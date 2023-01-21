import argparse
from pathlib import Path

import numpy
import resampy
import soundfile as sf
import torch
from tqdm import tqdm

from collections import OrderedDict
from eval import normalize_estimates_by_mse, load_model

def separate(
    audio,
    model,
    voice_parts,
    device='cpu',
    segment_length=None
):
    """Separate mixture audio signal

    Args:
        audio (numpy.ndarray): Mixture audio (n_channels x time)
        model (nn.Module): Trained model
        voice_parts (list): List of voice parts, e.g., ["soprano", "alto", "tenor"]
        device (str, optional): Device for torch (defaults: `cpu`)
        segment_length (int, optional): Segment length to be processed at a time (If `None`, it equals the signal length.)

    Returns:
        dict[str,numpy.ndarray]: Dictionary of all estimates obtained with the model (time x n_channels)
    """
    
    estimates = OrderedDict()
    for j, name in enumerate(voice_parts):
        estimates[name] = []

    for c in range(audio.shape[0]):
        audio_torch = torch.tensor(audio[c:c+1,None,:]).float().to(device) # 1 x 1 x time
        if segment_length is None:
            est_targets = model(audio_torch)[0,:,:].cpu().numpy() # n_sources x time        
        else:
            est_targets = numpy.zeros((len(voice_parts), audio_torch.shape[-1]))
            sp = 0
            cnt = 0
            while 1:
                print(cnt, sp, segment_length, audio_torch.shape[-1])
                if sp+segment_length > audio_torch.shape[-1]:
                    over_samples = sp+segment_length - audio_torch.shape[-1]
                    sp -= over_samples
                audio_segment_torch = audio_torch[:,:,sp:sp+segment_length]
                est_targets[:, sp:sp+segment_length] = model(audio_segment_torch)[0,:,:].cpu().numpy() # n_sources x time
                sp += segment_length
                cnt += 1
                if sp == audio_torch.shape[-1]:
                    break
        for j, name in enumerate(voice_parts):
            estimates[name].append(est_targets[j,:])

    for j, name in enumerate(voice_parts):
        estimates[name] = numpy.stack(estimates[name], axis=0).T # n_channels x time -> time x n_channels

    return estimates

def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument('--segment', default=None, type=float, help="Processed segment length in seconds")

    return inf_parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', default=None)
    parser.add_argument('--start', type=float, default=0.0, help='Audio chunk start in seconds')
    parser.add_argument('--duration', type=float, default=-1.0, help='Audio chunk duration in seconds, negative values load full track')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for separated signals')
    parser.add_argument('--no_normalization', action='store_true')
    parser.add_argument('input_files', type=str, nargs="*", help="To-be-separated audio file paths")
    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    if args.model_dir is None:
        model_path = None
    else:
        model_path = Path(args.model_dir) / 'best_model.pth'
        if not model_path.exists():
            raise ValueError(f'Model file not found [{model_path}]')

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    out_root_dir = Path(args.output_dir)

    # load model
    model, sources, _, _ = load_model(model_path, device)

    # Stack to-be-separated audio files
    input_files = [Path(_) for _ in args.input_files]
    
    # main loop
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
                audio.T, # n_channels x time
                model,
                sources,
                device=device,
                segment_length=None if args.segment is None else int(model.sample_rate*args.segment)
            )
            if not args.no_normalization:
                stacked_estimates = numpy.stack([estimates[name].T for name in sources], axis=0) # n_sources x n_channels x time
                stacked_estimates = normalize_estimates_by_mse(stacked_estimates, audio.T)
                for src_idx, name in enumerate(sources):
                    estimates[name] = stacked_estimates[src_idx,:,:].T            

        # save
        for target, estimate in estimates.items():
            if org_samplerate != model.sample_rate:
                estimate = resampy.resample(estimate, model.sample_rate, org_samplerate, axis=0)
            sf.write(output_dir / f'{target}.wav', estimate, org_samplerate, subtype=info.subtype)
