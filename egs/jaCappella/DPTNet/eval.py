import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy
import pandas
import torch
import yaml
from asteroid.metrics import get_metrics
from asteroid.models import DPTNet
from tqdm import tqdm

from local import dataloader

def normalize_estimates_by_mse(estimates, mixture):
    '''
    Args:
        estimates: source x channel x time
        mixture: channel x time
    '''
    separation = []
    for c in range(estimates.shape[1]):
        a_l = estimates[:,c,:].T # source x time -> time x source
        b_l = mixture[c, :] # time
        sol_l = numpy.linalg.lstsq(a_l, b_l, rcond=None)[0] # inst
        e_l = a_l * sol_l
        separation.append(e_l)
    separation = numpy.stack(separation, axis=0) # channel x time x source
    return numpy.transpose(separation, (2,0,1)) # source x channel x time

@dataclass
class DummyArgs:
    test_dir: Path
    sources: List[str]

def evaluate_for_pit_trained_model(estimates: numpy.ndarray, targets: numpy.ndarray, mix: numpy.ndarray, sample_rate: float=48000):
    '''Evaluate the source estimates

    Args:
        estimates (numpy.ndarray): Source estimates (n_sources x n_channels x time)
        targets (numpy.ndarray): Groundtruths (n_sources x n_channels x time)
        mix (numpy.ndarray): Input mixture (n_channels x time)

    Return:
        numpy.ndarray: SI-SDR improvements [dB] averaged over channel (n_sources)
    '''
    estimates = estimates.astype(numpy.float64)
    targets = targets.astype(numpy.float64)
    mix = mix.astype(numpy.float64)
    #
    n_sources, n_channels, _ = estimates.shape
    best_sisdr_improvements_list = []
    best_sisdr_list = []
    best_input_sisdr_list = []
    for c in range(n_channels):
        metrics = get_metrics(mix[c,:], targets[:,c,:], estimates[:,c,:], sample_rate=sample_rate, compute_permutation=True, average=False, metrics_list=["si_sdr"])
        sisdr, input_sisdr = metrics["si_sdr"], metrics["input_si_sdr"]
        sisdr = sisdr.reshape(n_sources)
        input_sisdr = input_sisdr.reshape(n_sources)
        best_sisdr_imp = sisdr - input_sisdr
        best_sisdr_improvements_list.append(best_sisdr_imp) # sources x 1
        best_sisdr_list.append(sisdr)
        best_input_sisdr_list.append(input_sisdr)
    sisdr_improvements = numpy.stack(best_sisdr_improvements_list, axis=1) # sources x channel
    sisdr_improvements = sisdr_improvements.mean(axis=1)
    sisdr = numpy.stack(best_sisdr_list, axis=1).mean(axis=1)
    input_sisdr = numpy.stack(best_input_sisdr_list, axis=1).mean(axis=1)
    return sisdr_improvements, sisdr, input_sisdr

def evaluate(estimates: numpy.ndarray, targets: numpy.ndarray, mix: numpy.ndarray, sample_rate: float=48000):
    '''Evaluate the source estimates

    Args:
        estimates (numpy.ndarray): Source estimates (n_sources x n_channels x time)
        targets (numpy.ndarray): Groundtruths (n_sources x n_channels x time)
        mix (numpy.ndarray): Input mixture (n_channels x time)

    Return:
        numpy.ndarray: SI-SDR improvements [dB] averaged over channel (n_sources)
    '''
    estimates = estimates.astype(numpy.float64)
    targets = targets.astype(numpy.float64)
    mix = mix.astype(numpy.float64)
    #
    n_sources, n_channels, time_length = estimates.shape
    best_sisdr_improvements_list = []
    sisdr_list = []
    input_sisdr_list = []
    for c in range(n_channels):
        metrics = get_metrics(mix[c,:], targets[:,c,:], estimates[:,c,:], sample_rate=sample_rate, compute_permutation=False, average=False, metrics_list=["si_sdr"])
        sisdr, input_sisdr = metrics["si_sdr"], metrics["input_si_sdr"]
        sisdr = sisdr.reshape(n_sources)
        input_sisdr = input_sisdr.reshape(n_sources)
        best_sisdr_imp = sisdr - input_sisdr
        best_sisdr_improvements_list.append(best_sisdr_imp) # sources x 1
        sisdr_list.append(sisdr)
        input_sisdr_list.append(input_sisdr)
    sisdr_improvements = numpy.stack(best_sisdr_improvements_list, axis=1) # sources x channel
    sisdr_improvements = sisdr_improvements.mean(axis=1)
    sisdr = numpy.stack(sisdr_list, axis=1).mean(axis=1)
    input_sisdr = numpy.stack(input_sisdr_list, axis=1).mean(axis=1)
    return sisdr_improvements, sisdr, input_sisdr

def load_model(model_name, device='cpu'):
    model = DPTNet.from_pretrained(str(model_name))
    model.eval()
    model.to(device)
    with open(model_name.parent / "conf.yml", "r") as fp:
        conf = yaml.safe_load(fp)
    return model, conf["data"]["sources"], conf["data"]["seq_dur"], conf

def separate_for_pit_trained_model(
    audio,
    model,
    voice_parts,
    device='cpu',
    segment_length=None,
):
    """Separate mixture audio signal

    Args:
        audio (numpy.ndarray): Mixture audio (n_channels x time)
        model (nn.Module): Trained model
        voice_parts (list): List of voice parts, e.g., ["soprano", "alto", "tenor"]
        device (str, optional): Device for torch (defaults: `cpu`)
        segment_length (int, optional): Segment length to be processed at a time (If `None`, it equals the signal length.)

    Returns:
        numpy.ndarray: All estimates obtained with the model (n_sources x n_channels x time)
    """
    
    est_targets_list = []
    for c in range(audio.shape[0]):
        audio_torch = torch.tensor(audio[c:c+1,None,:]).clone().detach().float().to(device) # 1 x 1 x time
        if segment_length is None or segment_length > audio_torch.shape[-1]:
            est_targets = model(audio_torch)[0,:,:].cpu().numpy() # n_sources x time        
        else:
            est_targets = numpy.zeros((len(voice_parts), audio_torch.shape[-1]))
            for sp in tqdm(range(0, audio_torch.shape[-1], segment_length), desc="Process segments", leave=False):
                if sp+segment_length > audio_torch.shape[-1]:
                    over_samples = sp+segment_length - audio_torch.shape[-1]
                    sp -= over_samples
                audio_segment_torch = audio_torch[:,:,sp:sp+segment_length]
                est_targets[:, sp:sp+segment_length] = model(audio_segment_torch)[0,:,:].cpu().numpy() # n_sources x time
        est_targets_list.append(est_targets)
    estimates = numpy.stack(est_targets_list, axis=1) # n_sources x n_channels x time
    return estimates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--no_normalization', action='store_true')
    args = parser.parse_args()

    model_path = Path(args.model_dir) / 'best_model.pth'
    if not model_path.exists():
        raise ValueError(f'Model file not found [{model_path}]')

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_summary_filename = Path(args.model_dir) / "eval_results.csv"

    # load model
    model, sources, val_dur, conf = load_model(model_path, device=device)
        
    # load dataset
    test_dataset = dataloader.load_test_dataset(parser, DummyArgs(test_dir=args.test_dir, sources=sources))

    with open(eval_summary_filename, "w") as fp:
        print('song_name,target,sisdr_imp,sisdr,input_sisdr', file=fp)
        for data_index in tqdm(range(len(test_dataset))):
            track_path = test_dataset.get_track_path(data_index)
            mix, gts = test_dataset[data_index]
            with torch.inference_mode():
                estimates = separate_for_pit_trained_model(
                    mix, # n_channels x time
                    model,
                    sources,
                    device=device,
                    segment_length=int(round(val_dur*model.sample_rate))
                ) # source x channel x time
                #####
                if not args.no_normalization:
                    estimates = normalize_estimates_by_mse(estimates, mix)
                #####
                gts = gts.numpy()
                sisdr_improvements, sisdrs, input_sisdrs = evaluate_for_pit_trained_model(estimates, gts, mix.numpy(), sample_rate=test_dataset.sample_rate)
                for i, source_name in enumerate(sources):
                    print(f'{track_path.name},{source_name},{float(sisdr_improvements[i])},{float(sisdrs[i])},{float(input_sisdrs[i])}', file=fp)
    #####
    summary = pandas.read_csv(eval_summary_filename)
    print(summary.groupby("target")["sisdr_imp"].describe().round(1))

