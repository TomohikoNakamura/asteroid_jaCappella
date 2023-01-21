import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import huggingface_hub
import numpy
import pandas
import torch
import yaml
from local import dataloader
from mrdla import MRDLA
from tqdm import tqdm

from asteroid.metrics import get_metrics

@dataclass
class DummyArgs:
    test_dir: Path
    sources: List[str]
    in_memory: bool=False

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
    return sisdr_improvements, sisdrs, input_sisdrs
    

def load_model(model_name: Path, device='cpu'):
    if model_name is None:
        model_name = huggingface_hub.hf_hub_download(
            repo_id="tnkmr/MRDLA_jaCappella_VES_48k",
            filename="best_model.pth",
            cache_dir="pretrained",
        )
        model = MRDLA.from_pretrained(str(model_name))
        conf_name = huggingface_hub.hf_hub_download(
            repo_id="tnkmr/MRDLA_jaCappella_VES_48k",
            filename="conf.yml",
            cache_dir="pretrained",
        )
        with open(conf_name, "r") as fp:
            conf = yaml.safe_load(fp)
    else:
        model = MRDLA.from_pretrained(str(model_name))
        with open(model_name.parent / "conf.yml", "r") as fp:
            conf = yaml.safe_load(fp)
    model.eval()
    model.to(device)
    return model, conf["data"]["sources"]

def separate(
    audio: torch.Tensor,
    mrdla_model: MRDLA,
    instruments: List[str],
    device='cpu',
    shift_length=None
):
    source_names = [_ for _ in instruments]
    estimates = {name: [] for name in source_names}
    for c in range(audio.shape[0]): # channel loop
        # convert numpy audio to torch
        audio_torch = torch.tensor(audio[c:c+1,None,:]).clone().detach().float().to(device) # 1 x 1 x nb_timesteps
        if shift_length:
            est_targets = mrdla_model.sequential_predict_w_shifts(audio_torch, shift_length=shift_length).cpu()[0,...].numpy() # sources x 1 x time
        else:
            est_targets = mrdla_model(audio_torch).cpu()[0,...].numpy() # sources x 1 x time
        for j, name in enumerate(source_names):
            estimates[name].append(est_targets[j,:,:]) # 1 x time
            
    for j, name in enumerate(source_names):
        estimates[name] = numpy.concatenate(estimates[name], axis=0) # ch x time

    return estimates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--model_dir', type=str, help='Results path where ' 'best_model.pth' ' is stored', required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    args = parser.parse_args()

    model_path = Path(args.model_dir) / 'best_model.pth'
    if not model_path.exists():
        raise ValueError(f'Model file not found [{model_path}]')

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_summary_filename = Path(args.model_dir) / "eval_results.csv"

    # load model
    model, sources = load_model(model_path, device=device)
    assert model.signal_ch==1, f'Supported only monaural model'

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

