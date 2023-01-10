import argparse
from pathlib import Path

import torch
import resampy
from tqdm import tqdm
import soundfile as sf

from eval import load_model, separate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', default=None, required=True)
    parser.add_argument('--start', type=float, default=0.0, help='Audio chunk start in seconds')
    parser.add_argument('--duration', type=float, default=-1.0, help='Audio chunk duration in seconds, negative values load full track')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for separated signals')
    parser.add_argument('--shift_length', type=float, default=None)
    parser.add_argument('input_files', type=str, nargs="*")
    args = parser.parse_args()

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
    model, sources = load_model(model_path, device)

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
                device=device,
                shift_length=int(round(args.shift_length*model.sample_rate)) if args.shift_length else None
            )

        # save
        for target, estimate in estimates.items():
            estimate = estimate.T
            if org_samplerate != model.sample_rate:
                estimate = resampy.resample(estimate, model.sample_rate, org_samplerate, axis=0)
            sf.write(output_dir / f'{target}.wav', estimate, org_samplerate, subtype=info.subtype)
