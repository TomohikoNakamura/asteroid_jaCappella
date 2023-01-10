import argparse
import sys
from pathlib import Path

import numpy as np
import norbert
import resampy
import scipy.signal
import soundfile as sf
import torch
import yaml
from asteroid.complex_nn import torch_complex_from_magphase
from asteroid.data import jaCappellaCorpus
from asteroid.models import XUMX, x_umx
from tqdm import tqdm

from train import define_model
from collections import OrderedDict


def _load_model_from_ckpt(model_path: Path, device='cpu'):
    with open(model_path.parent.parent / 'conf.yml', 'r') as fp:
        conf = yaml.load(fp, Loader=yaml.SafeLoader)

    model = define_model(
        scaler_mean=None,
        scaler_std=None,
        sample_rate=conf['data']['sample_rate'],
        in_chan=conf['model']['in_chan'],
        bandwidth=conf['model']['bandwidth'],
        window_length=conf['model']['window_length'],
        nb_channels=conf['model']['nb_channels'],
        hidden_size=conf['model']['hidden_size'],
        nhop=conf['model']['nhop'],
        sources=conf['data']['sources'],
        bidirectional=conf['model']['bidirectional'],
        spec_power=conf['model']['spec_power'],
        loss_use_multidomain=conf['training']['loss_use_multidomain']
    )
    _weights = torch.load(model_path, map_location=device)["state_dict"]
    weights = OrderedDict()
    for k, v in _weights.items():
        if "model." in k:
            weights[k.replace("model.", "")] = v
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model, model.sources

def _load_model(model_name, device='cpu'):
    print('Loading model from: {}'.format(model_name), file=sys.stderr)
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
    audio: np.ndarray [shape=(nb_channels, nb_timesteps)]
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
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    V = []
    X = None
    source_names = [_ for _ in instruments]
    for c in range(audio.shape[0]): # channel loop
        # convert numpy audio to torch
        audio_torch = torch.tensor(audio[c:c+1,None,:]).float().to(device) # 1 x 1 x nb_timesteps
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
                V[j] = np.concatenate((V[j], Vj[:,0,Ellipsis]), axis=1)
        #######
        tmp = x_umx_target.encoder(audio_torch)
        Xc = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
        Xc = Xc.detach().cpu().numpy()
        Xc = Xc[0].transpose(2, 1, 0)
        if c==0:
            X = Xc # nb_frames, nb_bins, nb_channels
        else:
            X = np.concatenate((X,Xc), axis=2)

    V = np.transpose(np.array(V), (1, 3, 2, 0))
    # print(V.shape, X.shape)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            # length=audio.shape[-1], # nb_samples, nb_channels, nb_timesteps
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        # print(audio_hat.shape)
        estimates[name] = audio_hat.T

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

def load_model(args, device):
    if args.model_dir is not None:
        model_path = Path(args.model_dir) / 'best_model.pth'
        load_func = _load_model
    else:
        model_path = Path(args.ckpt)
        load_func = _load_model_from_ckpt
    if not model_path.exists():
        raise ValueError(f'Model file not found [{model_path}]')
    model, sources = load_func(model_path, device=device)
    return model, sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument('--ckpt', type=str, help='checkpoint', default=None)
    arg_group.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', default=None)
    parser.add_argument('--start', type=float, default=0.0, help='Audio chunk start in seconds')
    parser.add_argument('--duration', type=float, default=-1.0, help='Audio chunk duration in seconds, negative values load full track')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for separated signals')
    parser.add_argument('input_files', type=str, nargs="*")
    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    if args.model_dir is not None:
        model_path = Path(args.model_dir) / 'best_model.pth'
    else:
        model_path = Path(args.ckpt)
    if not model_path.exists():
        raise ValueError(f'Model file not found [{model_path}]')

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    out_root_dir = Path(args.output_dir)

    # load model
    model, sources = load_model(args, device)
    assert model.nb_channels==1, f'Supported only monaural model'

    #
    input_files = [Path(_) for _ in args.input_files]

    #
    for input_file in tqdm(input_files):
        output_dir = out_root_dir / f'{input_file.parent.stem}_{input_file.stem}'
        output_dir.mkdir(exist_ok=True, parents=True)

        # handling an input audio path
        info = sf.info(input_file)
        start = int(args.start * info.samplerate)

        # check if dur is none
        if args.duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(args.duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None
        audio, org_samplerate = sf.read(input_file, always_2d=True, start=start, stop=stop) # time x ch
        if org_samplerate != model.sample_rate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, org_samplerate, model.sample_rate, axis=0)

        # separate
        with torch.inference_mode():
            estimates = separate(
                audio.T, # nb_channels x nb_timesteps
                model,
                sources,
                niter=args.niter,
                alpha=args.alpha,
                softmask=args.softmask,
                residual_model=args.residual_model,
                device=device,
            )

        # save
        for target, estimate in estimates.items():
            if org_samplerate != model.sample_rate:
                estimate = resampy.resample(estimate, model.sample_rate, org_samplerate, axis=0)
            sf.write(output_dir / f'{target}.wav', estimate, org_samplerate, subtype=info.subtype)
