import argparse
import copy
import json
import os
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.data
from asteroid import DPTNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.schedulers import DPTNetScheduler
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, multisrc_neg_sisdr
from asteroid.metrics import get_metrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from local import dataloader

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
# parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
# parser.add_argument("--train_json", default="exp/tmp", help="Full path to save best validation model")
# parser.add_argument("--val_json", default="exp/tmp", help="Full path to save best validation model")

class AugSystem(System):
    default_monitor: str = "val_loss"
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None
    ):
        self.sources = copy.deepcopy(config["data"]["sources"])
        config["data"].pop("sources")
        config["data"].pop("source_augmentations")
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)
        self.val_dur_samples = int(model.sample_rate * config["data"]["seq_dur"])
        if "pit" in self.config["training"]["loss_func"]:
            self.use_pit = True
        else:
            self.use_pit = False
    
    def common_step(self, batch, batch_nb, train=True):
        mix, targets = batch # batch x channel(1) x time, batch x sources x channel(1) x time
        assert mix.shape[1] == 1 and targets.shape[2] == 1
        targets = targets[:,:,0,:] # batch x sources x time
        est_targets = self(mix)
        if train:
            loss = self.loss_func(est_targets, targets)
            return loss
        else:
            if self.use_pit:
                loss, reordered_est_targets = self.loss_func(est_targets, targets, return_est=True)
                return loss.mean(), reordered_est_targets
            else:
                loss = self.loss_func(est_targets, targets)
                return loss.mean(), est_targets

    def _sequential_common_step(self, batch, batch_nb):
        sp = 0
        dur_samples = int(self.val_dur_samples)
        cnt = 0
        loss_tmp = 0.0
        reordered_est_targets = []

        while 1:
            batch_tmp = [
                batch[0][Ellipsis, sp : sp + dur_samples],
                batch[1][Ellipsis, sp : sp + dur_samples],
            ]
            _loss_tmp, reordered_est_targets_tmp = self.common_step(batch_tmp, batch_nb, train=False)
            loss_tmp += _loss_tmp
            reordered_est_targets.append(reordered_est_targets_tmp) # batch x sources x time_length
            cnt += 1
            sp += dur_samples
            if batch_tmp[0].shape[-1] < dur_samples or batch[0].shape[-1] == cnt * dur_samples:
                break
        loss = loss_tmp / cnt
        # loss, reordered_est_targets = self.common_step(batch, batch_nb, train=False)
        reordered_est_targets = torch.cat(reordered_est_targets, dim=reordered_est_targets[0].ndim-1)
        return loss, reordered_est_targets

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        """
        We calculate the ``validation loss'' by splitting each song into
        smaller chunks in order to prevent GPU out-of-memory errors.
        The length of each chunk is given by ``self.val_dur_samples'' which is
        computed from ``sample_rate [Hz]'' and ``val_dur [seconds]'' which are
        both set in conf.yml.
        """
        tag = "val"
        #
        inputs = batch[0].detach()
        targets = batch[1].detach()
        assert inputs.shape[0] == 1 and targets.shape[0] == 1, f'Validation loop supported only for single batch'
        #
        loss, reordered_est_targets = self._sequential_common_step(batch, batch_nb)
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        #
        # inputs = inputs.reshape(-1, inputs.shape[-1]) # 1 x time_length
        # targets = targets.reshape(targets.shape[1], -1) # 1 x time_length
        # reordered_est_targets = reordered_est_targets.reshape(reordered_est_targets.shape[1], -1) # 1 x time_length
        # metrics = get_metrics(inputs.cpu().numpy(), targets.cpu().numpy(), reordered_est_targets.cpu().numpy(), sample_rate=self.config["data"]["sample_rate"], metrics_list=["si_sdr"], average=False)
        # val_sisdr = metrics["si_sdr"].reshape(-1) - metrics["input_si_sdr"].reshape(-1)
        # self.log(f"{tag}_sisdr/average", float(val_sisdr.mean()), on_epoch=True, prog_bar=True, sync_dist=True)
        # for i, source in enumerate(self.sources):
        #     self.log(f"{tag}_sisdr/{source}", float(val_sisdr[i]), on_epoch=True, prog_bar=False, sync_dist=True)

def main(conf, args):
    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["OMP_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"
    os.environ["MKL_NUM_THREADS"] = f"{int(args.num_workers/4*3)}"

    # create output dir if not exist
    exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Define Dataloader
    train_dataset, valid_dataset = dataloader.load_datasets(parser, args)
    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    )
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    dataloader_kwargs["num_workers"] = 1
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Load model
    model = DPTNet(**conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"])
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    schedulers = {
        "scheduler": DPTNetScheduler(
            optimizer, len(train_sampler) // conf["training"]["batch_size"], conf["masknet"]["in_chan"] # 64
        ),
        "interval": "step",
    }

    # Just after instantiating, save the args. Easy loading in the future.
    with open(exp_dir / "conf.yml", "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    if (not hasattr(args, "loss_func")) or (args.loss_func == "pit_sisdr"):
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    elif args.loss_func == "sisdr":
        loss_func = multisrc_neg_sisdr
    else:
        raise ValueError
    system = AugSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=schedulers,
        train_loader=train_sampler,
        val_loader=valid_sampler,
        config=conf,
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
    es = EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True)
    callbacks.append(es)

    # Don't ask GPU if they are not available.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy='ddp',
        devices='auto',
        gradient_clip_val=conf["training"]["gradient_clipping"],
        limit_train_batches=1.0,
        check_val_every_n_epoch=1,
        val_check_interval=None
    )
    if args.ckpt_path is not None:
        trainer.fit(system, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(system)

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
