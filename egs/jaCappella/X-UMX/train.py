import argparse
import copy
import itertools
import json
from multiprocessing.sharedctypes import Value
import os
import random
from operator import itemgetter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import sklearn.preprocessing
import torch
import torch.utils.data
import tqdm
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import singlesrc_mse
from asteroid.metrics import get_metrics
from asteroid.models import XUMX
from asteroid.models.x_umx import _STFT, _Spectrogram
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.nn.modules.loss import _Loss

from local import dataloader

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs.
parser = argparse.ArgumentParser()

def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1

def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        _STFT(window_length=args.window_length, n_fft=args.in_chan, n_hop=args.nhop),
        _Spectrogram(spec_power=args.spec_power, mono=True),
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.random_segments = False
    dataset_scaler.random_track_mix = False
    dataset_scaler.segment = False
    pbar = tqdm.tqdm(range(len(dataset_scaler)))
    for ind in pbar:
        x, _ = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])[0]
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std

def freq_domain_loss(s_hat, gt_spec, combination=True):
    """Calculate frequency-domain loss between estimated and reference spectrograms.
    MSE between target and estimated target spectrograms is adopted as frequency-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        s_hat: estimated spectrograms
            (Sources, Freq. bins, Batch size, Channels, Frames)
        gt_spec: reference spectrograms
            (Freq. bins, Batch size, Sources x Channels, Frames)
        combination: whether use combination or not (optional)
    Output:
        calculated frequency-domain loss
    """

    n_src = len(s_hat)
    idx_list = [i for i in range(n_src)]

    inferences = []
    refrences = []
    for i, s in enumerate(s_hat):
        n_channels = s.shape[-2]
        inferences.append(s)
        refrences.append(gt_spec[..., n_channels * i : n_channels * i + n_channels, :])
    assert inferences[0].shape == refrences[0].shape

    _loss_mse = 0.0
    cnt = 0.0
    for i in range(n_src):
        _loss_mse += singlesrc_mse(inferences[i], refrences[i]).mean()
        cnt += 1.0

    # If Combination is True, calculate the expected combinations.
    if combination:
        for c in range(2, n_src):
            patterns = list(itertools.combinations(idx_list, c))
            for indices in patterns:
                tmp_loss = singlesrc_mse(
                    sum(itemgetter(*indices)(inferences)),
                    sum(itemgetter(*indices)(refrences)),
                ).mean()
                _loss_mse += tmp_loss
                cnt += 1.0

    _loss_mse /= cnt

    return _loss_mse

def time_domain_loss(mix, time_hat, gt_time, combination=True):
    """Calculate weighted time-domain loss between estimated and reference time signals.
    weighted SDR [1] between target and estimated target signals is adopted as time-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        mix: mixture time signal
            (Batch size, Channels, Time Length (samples))
        time_hat: estimated time signals
            (Sources, Batch size, Channels, Time Length (samples))
        gt_time: reference time signals
            (Batch size, Sources x Channels, Time Length (samples))
        whether use combination or not (optional)
    Output:
        calculated time-domain loss

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    n_src, n_batch, n_channel, time_length = time_hat.shape
    idx_list = [i for i in range(n_src)]

    # Fix Length
    mix = mix[Ellipsis, :time_length]
    gt_time = gt_time[Ellipsis, :time_length]

    # Prepare Data and Fix Shape
    mix_ref = [mix]
    mix_ref.extend([gt_time[..., n_channel * i : n_channel * i + n_channel, :] for i in range(n_src)])
    mix_ref = torch.stack(mix_ref)
    mix_ref = mix_ref.view(-1, time_length)
    time_hat = time_hat.view(n_batch * n_channel * time_hat.shape[0], time_hat.shape[-1])

    # If Combination is True, calculate the expected combinations.
    if combination:
        indices = []
        for c in range(2, n_src):
            indices.extend(list(itertools.combinations(idx_list, c)))

        for tr in indices:
            sp = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]
            ep = [n_batch * n_channel * (tr[i] + 2) for i in range(len(tr))]
            spi = [n_batch * n_channel * tr[i] for i in range(len(tr))]
            epi = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]

            tmp = sum([mix_ref[sp[i] : ep[i], ...].clone() for i in range(len(tr))])
            tmpi = sum([time_hat[spi[i] : epi[i], ...].clone() for i in range(len(tr))])
            mix_ref = torch.cat([mix_ref, tmp], dim=0)
            time_hat = torch.cat([time_hat, tmpi], dim=0)

        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src + len(indices), 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]
    else:
        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src, 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]

    # Calculation
    _loss_sdr = weighted_sdr(time_hat, refrences_t, mix_t)

    return 1.0 + _loss_sdr

def weighted_sdr(input, gt, mix, weighted=True, eps=1e-10):
    # ``input'', ``gt'' and ``mix'' should be (Batch, Time Length)
    assert input.shape == gt.shape
    assert mix.shape == gt.shape

    ns = mix - gt
    ns_hat = mix - input

    if weighted:
        alpha_num = (gt * gt).sum(1, keepdims=True)
        alpha_denom = (gt * gt).sum(1, keepdims=True) + (ns * ns).sum(1, keepdims=True)
        alpha = alpha_num / (alpha_denom + eps)
    else:
        alpha = 0.5

    # Target
    num_cln = (input * gt).sum(1, keepdims=True)
    denom_cln = torch.sqrt(eps + (input * input).sum(1, keepdims=True)) * torch.sqrt(
        eps + (gt * gt).sum(1, keepdims=True)
    )
    sdr_cln = num_cln / (denom_cln + eps)

    # Noise
    num_noise = (ns * ns_hat).sum(1, keepdims=True)
    denom_noise = torch.sqrt(eps + (ns_hat * ns_hat).sum(1, keepdims=True)) * torch.sqrt(
        eps + (ns * ns).sum(1, keepdims=True)
    )
    sdr_noise = num_noise / (denom_noise + eps)

    return torch.mean(-alpha * sdr_cln - (1.0 - alpha) * sdr_noise)

class MultiDomainLoss(_Loss):
    """A class for calculating loss functions of X-UMX.

    Args:
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        spec_power (int): Exponent for spectrogram calculation.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        loss_combine_sources (bool): Set to true if you are using the combination scheme
            proposed in [1]. If you select ``loss_combine_sources: yes'' via
            conf.yml, this is set as True.
        loss_use_multidomain (bool): Set to true if you are using a frequency- and time-domain
            losses collaboratively, i.e., Multi Domain Loss (MDL) proposed in [1].
            If you select ``loss_use_multidomain: yes'' via conf.yml, this is set as True.
        mix_coef (float): A mixing parameter for multi domain losses

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        window_length,
        in_chan,
        n_hop,
        spec_power,
        nb_channels,
        loss_combine_sources,
        loss_use_multidomain,
        mix_coef,
    ):
        super().__init__()
        self.transform = nn.Sequential(
            _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop),
            _Spectrogram(spec_power=spec_power, mono=False), # (nb_channels == 1)),
        )
        self._combi = loss_combine_sources
        self._multi = loss_use_multidomain
        self.coef = mix_coef
        print("Combination Loss: {}".format(self._combi))
        if self._multi:
            print(
                "Multi Domain Loss: {}, scaling parameter for time-domain loss={}".format(
                    self._multi, self.coef
                )
            )
        else:
            print("Multi Domain Loss: {}".format(self._multi))
        self.cnt = 0

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)

        targets: (Batch, Source, Channels, TimeLen)
        """

        spec_hat = est_targets[0]
        time_hat = est_targets[1]

        # Fix shape and apply transformation of targets
        n_batch, n_src, n_channel, time_length = targets.shape
        targets = targets.view(n_batch, n_src * n_channel, time_length)
        Y = self.transform(targets)[0]

        if self._multi:
            n_src = spec_hat.shape[0]
            mixture_t = sum([targets[:, n_channel * i : n_channel * i + n_channel, ...] for i in range(n_src)])
            loss_f = freq_domain_loss(spec_hat, Y, combination=self._combi)
            loss_t = time_domain_loss(mixture_t, time_hat, targets, combination=self._combi)
            loss = float(self.coef) * loss_t + loss_f
        else:
            loss = freq_domain_loss(spec_hat, Y, combination=self._combi)

        return loss

class XUMXManager(System):
    """A class for X-UMX systems inheriting the base system class of lightning.
    The difference from base class is specialized for X-UMX to calculate
    validation loss preventing the GPU memory over flow.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.dataloader): Training dataloader.
        val_loader (torch.utils.data.dataloader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
        val_dur (float): When calculating validation loss, the loss is calculated
            per this ``val_dur'' in seconds on GPU to prevent memory overflow.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        val_dur=None,
    ):
        self.sources = copy.deepcopy(config["data"]["sources"])
        config["data"].pop("sources")
        config["data"].pop("source_augmentations")
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)
        #
        self.val_dur_samples = model.sample_rate * val_dur

    def validation_step(self, batch, batch_nb, dataloader_idx):
        """
        We calculate the ``validation loss'' by splitting each song into
        smaller chunks in order to prevent GPU out-of-memory errors.
        The length of each chunk is given by ``self.val_dur_samples'' which is
        computed from ``sample_rate [Hz]'' and ``val_dur [seconds]'' which are
        both set in conf.yml.
        """
        tag = "val"
        _, targets = batch[0], batch[1] # targets: batch x sources x channel x time
        inputs = targets.sum(dim=1) # batch x sources x channels
        est_targets = self(inputs) # sources, batch_size, channels, time_length
        loss = self.loss_func(est_targets, targets) 
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        #
        time_hat = est_targets[1] # source x batch x channel x time
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(targets.shape[1], -1)
        time_hat = time_hat.reshape(time_hat.shape[0], -1)
        # print(inputs.shape, targets.shape, time_hat.shape)
        metrics = get_metrics(inputs.cpu().numpy(), targets.cpu().numpy(), time_hat.cpu().numpy(), sample_rate=self.config["data"]["sample_rate"], metrics_list=["si_sdr"], average=False)
        val_sisdr = metrics["si_sdr"].reshape(-1) - metrics["input_si_sdr"].reshape(-1)
        self.log(f"{tag}_sisdr/average", float(val_sisdr.mean()), on_epoch=True, prog_bar=True, sync_dist=True)
        for i, source in enumerate(self.sources):
            self.log(f"{tag}_sisdr/{source}", float(val_sisdr[i]), on_epoch=True, prog_bar=False, sync_dist=True)



def define_model(scaler_mean, scaler_std, sample_rate, in_chan, bandwidth, window_length, nb_channels, hidden_size, nhop, sources, bidirectional, spec_power, loss_use_multidomain) -> XUMX:
    max_bin = bandwidth_to_max_bin(sample_rate, in_chan, bandwidth)
    x_unmix = XUMX(
        window_length=window_length,
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=nb_channels,
        hidden_size=hidden_size,
        in_chan=in_chan,
        n_hop=nhop,
        sources=sources,
        max_bin=max_bin,
        bidirectional=bidirectional,
        sample_rate=sample_rate,
        spec_power=spec_power,
        return_time_signals=True if loss_use_multidomain else False,
    )
    return x_unmix

def main(conf, args):
    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["OMP_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"
    os.environ["MKL_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"

    # create output dir if not exist
    exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    train_dataset, valid_dataset = dataloader.load_datasets(parser, args)
    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    )
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    dataloader_kwargs["num_workers"] = 1
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Define model and optimizer
    if args.pretrained is not None:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)
    x_unmix = define_model(
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        sample_rate=train_dataset.sample_rate,
        in_chan=args.in_chan,
        bandwidth=args.bandwidth,
        window_length=args.window_length,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        nhop=args.nhop,
        sources=args.sources,
        bidirectional=args.bidirectional,
        spec_power=args.spec_power,
        loss_use_multidomain=args.loss_use_multidomain
    )

    optimizer = make_optimizer(
        x_unmix.parameters(), lr=args.lr, optimizer="adam", weight_decay=args.weight_decay
    )

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, cooldown=10
    )

    # Save config
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True) # , check_on_train_epoch_end=True)

    # Define Loss function.
    loss_func = MultiDomainLoss(
        window_length=args.window_length,
        in_chan=args.in_chan,
        n_hop=args.nhop,
        spec_power=args.spec_power,
        nb_channels=args.nb_channels,
        loss_combine_sources=args.loss_combine_sources,
        loss_use_multidomain=args.loss_use_multidomain,
        mix_coef=args.mix_coef,
    )
    system = XUMXManager(
        model=x_unmix,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_sampler,
        val_loader=valid_sampler, # first sampler must be for validation data
        scheduler=scheduler,
        config=conf,
        val_dur=args.val_dur
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True,
        every_n_epochs=1, every_n_train_steps=None, train_time_interval=None,
        save_last=True
    )
    callbacks.append(checkpoint)
    callbacks.append(es)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        check_val_every_n_epoch=1,
        val_check_interval=None
    )
    if args.ckpt_path is None:
        trainer.fit(system)
    else:
        trainer.fit(system, ckpt_path=args.ckpt_path)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_dataset.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from asteroid.utils import parse_args_as_dict, prepare_parser_from_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, plain_args)
