import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import huggingface_hub
import norbert
import numpy
import pandas
import scipy.signal
import torch
from local import dataloader
from tqdm import tqdm

from asteroid.complex_nn import torch_complex_from_magphase
from asteroid.metrics import get_metrics
from asteroid.models import XUMX


@dataclass
class DummyArgs:
    test_dir: Path
    sources: List[str]

def evaluate(estimates: numpy.ndarray, targets: numpy.ndarray, mix: numpy.ndarray, sample_rate: float=48000):
    '''

    Args:
        estimates (numpy.ndarray): source x channel x time
        targets (numpy.ndarray): source x channel x time
        mix (numpy.ndarray): channel x time

    Return:
        SI-SDR improvements [dB] averaged over channel (sources)
    '''
    n_sources, n_channels, time_length = estimates.shape
    sisdrs = []
    input_sisdrs = []
    for c in range(n_channels):
        metrics = get_metrics(mix[c,:], targets[:,c,:], estimates[:,c,:], sample_rate=sample_rate, metrics_list=["si_sdr"], average=False)
        sisdrs.append(metrics["si_sdr"])
        input_sisdrs.append(metrics["input_si_sdr"])
    sisdrs = numpy.stack(sisdrs, axis=1)
    input_sisdrs = numpy.stack(input_sisdrs, axis=1)
    sisdr_improvements = sisdrs - input_sisdrs
    sisdr_improvements = sisdr_improvements.mean(axis=1)
    sisdrs = sisdrs.mean(axis=1)
    input_sisdrs = input_sisdrs.mean(axis=1)
    return sisdr_improvements, sisdrs, input_sisdrs
    

def load_model(model_name: Path, device='cpu'):
    if model_name is None:
        model_name = huggingface_hub.hf_hub_download(
            repo_id="tnkmr/XUMX_jaCappella_VES_48k",
            filename="best_model.pth",
            cache_dir="pretrained",
        )
    model = XUMX.from_pretrained(str(model_name))
    model.eval()
    model.to(device)
    return model, model.sources


def istft(X, rate=48000, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio

def separate(
    audio,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device='cpu',
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: numpy.ndarray [shape=(nb_channels, nb_timesteps)]
        mixture audio

    x_umx_target: asteroid.models
        X-UMX model used for separating

    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `numpy.ndarray`]
        dictionary with all estimates obtained by the separation model. (ch x time)
    """

    V = []
    X = None
    source_names = [_ for _ in instruments]
    for c in range(audio.shape[0]): # channel loop
        # convert numpy audio to torch
        audio_torch = torch.tensor(audio[c:c+1,None,:]).clone().detach().float().to(device) # 1 x 1 x nb_timesteps
        masked_tf_rep, _ = x_umx_target(audio_torch)
        for j, target in enumerate(instruments):
            Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy() # frames x 1(actual batch) x nb_channels x fbin
            if softmask:
                # only exponentiate the model if we use softmask
                Vj = Vj**alpha
            # output is nb_frames, nb_samples, nb_channels, nb_bins
            if c == 0:
                V.append(Vj[:, 0, Ellipsis])  # remove sample dim,  frames x nb_channels x fbin
            else:
                V[j] = numpy.concatenate((V[j], Vj[:,0,Ellipsis]), axis=1)
        #######
        tmp = x_umx_target.encoder(audio_torch)
        Xc = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
        Xc = Xc.detach().cpu().numpy()
        Xc = Xc[0].transpose(2, 1, 0)
        if c==0:
            X = Xc # nb_frames, nb_bins, nb_channels
        else:
            X = numpy.concatenate((X,Xc), axis=2)

    V = numpy.transpose(numpy.array(V), (1, 3, 2, 0))
    # print(V.shape, X.shape)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(numpy.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        if audio_hat.shape[-1] < audio.shape[-1]:
            rest = audio.shape[-1] - audio_hat.shape[-1]
            audio_hat = numpy.concatenate((audio_hat, numpy.zeros((audio_hat.shape[0], rest), dtype=audio_hat.dtype)), axis=1)
        estimates[name] = audio_hat # ch x time

    return estimates

def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=(
            'if enabled, will initialize separation with softmask.'
            'otherwise, will use mixture phase with spectrogram'
        ),
    )

    inf_parser.add_argument(
        '--niter', type=int, default=1, help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha', type=float, default=1.0, help='exponent in case of softmask separation'
    )

    inf_parser.add_argument('--samplerate', type=int, default=48000, help='model samplerate')

    inf_parser.add_argument(
        '--residual-model', action='store_true', help='create a model for the residual'
    )
    return inf_parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_dir', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    model_path = Path(args.model_dir) / 'best_model.pth'
    if not model_path.exists():
        raise ValueError(f'Model file not found [{model_path}]')

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_summary_filename = Path(args.model_dir) / "eval_results.csv"

    # load model
    model, sources = load_model(model_path, device=device)
    assert model.nb_channels==1, f'Supported only monaural model'

    # load dataset
    test_dataset = dataloader.load_test_dataset(parser, DummyArgs(test_dir=args.test_dir, sources=sources))

    with open(eval_summary_filename, "w") as fp:
        print('song_name,target,sisdr_imp,sisdr,input_sisdr', file=fp)
        for data_index in tqdm(range(len(test_dataset))):
            track_path = test_dataset.get_track_path(data_index)
            mix, gts = test_dataset[data_index]
            with torch.inference_mode():
                estimates_dict = separate(
                    mix, # nb_channels x nb_timesteps
                    model,
                    sources,
                    niter=args.niter,
                    alpha=args.alpha,
                    softmask=args.softmask,
                    residual_model=args.residual_model,
                    device=device,
                )
                #####
                estimates = []
                for source_name in sources:
                    estimates.append(estimates_dict[source_name])
                estimates = numpy.stack(estimates, axis=0) # source x channel x time
                #####
                gts = gts.numpy()
                sisdr_improvements, sisdrs, input_sisdrs = evaluate(estimates, gts, mix.numpy(), sample_rate=test_dataset.sample_rate)
                for i, source_name in enumerate(sources):
                    print(f'{track_path.name},{source_name},{float(sisdr_improvements[i])},{float(sisdrs[i])},{float(input_sisdrs[i])}', file=fp)
    #####
    summary = pandas.read_csv(eval_summary_filename)
    print(summary.groupby("target")["sisdr_imp"].describe().round(1))

