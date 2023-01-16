"""This code augments the data of the jaCappella corpus. It is based on demucs code (https://github.com/facebookresearch/demucs).
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script creates realistic mixes with stems from different songs.
In particular, it will align BPM, sync up the first beat and perform pitch
shift to maximize pitches overlap.
In order to limit artifacts, only parts that can be mixed with less than 15%
tempo shift, and 3 semitones of pitch shift are mixed together.
"""
import hashlib
import math
import os
import pickle
import random
import shutil
import subprocess
import tempfile
from collections import OrderedDict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import julius
import numpy
import pandas
import torch
import torchaudio
import tqdm
from librosa.beat import beat_track
from librosa.feature import chroma_cqt
from torch.nn import functional as F

# from dora.utils import try_load
# from demucs.audio import save_audio
# from demucs.repitch import repitch
# from demucs.pretrained import SOURCES
# from demucs.wav import build_metadata, Wavset

SOURCES = ["vocal_percussion", "bass", "alto", "tenor", "soprano", "lead_vocal"]

JACAPPELLA_PATH = Path("jaCappella_v1.0.0")
TEST_SONG_LIST_FILE_PATH = JACAPPELLA_PATH / "test_song_list_for_vocal_ensemble_separation.txt"
# WARNING: OUTPATH will be completely erased.
ORG_OUTPATH = Path("data") / 'org'
OUTPATH = Path("data") / 'augmented'
CACHE = Path("cache") / 'automix_cache'  # cache BPM and pitch information.
CHANNELS = 1
SR = 48000
MAX_PITCH = 3  # maximum allowable pitch shift in semi tones
MAX_TEMPO = 0.15  # maximum allowable tempo shift

#####################################
# from demucs.audio
def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav

def save_audio(wav, path, samplerate, bitrate=320, clip='rescale',
               bits_per_sample=16, as_float=False):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`.
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        raise NotImplementedError
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        torchaudio.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")

# from demucs.repitch
class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset, proba=0.2, max_pitch=2, max_tempo=12,
                 tempo_std=5, vocals=[3], same=True):
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.tempo_std = tempo_std
        self.same = same
        self.vocals = vocals

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int((1 - 0.01 * self.max_tempo) * in_length)

        if random.random() < self.proba:
            outs = []
            for idx, stream in enumerate(streams):
                if idx == 0 or not self.same:
                    delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
                    delta_tempo = random.gauss(0, self.tempo_std)
                    delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)
                stream = repitch(
                    stream,
                    delta_pitch,
                    delta_tempo,
                    voice=idx in self.vocals)
                outs.append(stream[:, :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams

def repitch(wav, pitch, tempo, voice=False, quick=False, samplerate=44100):
    """
    tempo is a relative delta in percentage, so tempo=10 means tempo at 110%!
    pitch is in semi tones.
    Requires `soundstretch` to be installed, see
    https://www.surina.net/soundtouch/soundstretch.html
    """
    infile = tempfile.NamedTemporaryFile(suffix=".wav")
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(wav, infile.name, samplerate, clip='clamp')
    command = [
        "soundstretch",
        infile.name,
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr = torchaudio.load(outfile.name)
    assert sr == samplerate
    return wav

# from demucs.audio
def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

# from demucs.wav
MIXTURE = "mixture"
EXT = ".wav"
def _track_metadata(track, sources, normalize=True, ext=EXT):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    for source in sources + [MIXTURE]:
        file = track / f"{source}{ext}"
        try:
            info = torchaudio.info(str(file))
        except RuntimeError:
            print(file)
            raise
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")
        if source == MIXTURE and normalize:
            try:
                wav, _ = torchaudio.load(str(file))
            except RuntimeError:
                print(file)
                raise
            wav = wav.mean(0)
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": length, "mean": mean, "std": std, "samplerate": track_samplerate}

def build_metadata(path, sources, normalize=True, ext=EXT):
    """
    Build the metadata for `Wavset`.

    Args:
        path (str or Path): path to dataset.
        sources (list[str]): list of sources to look for.
        normalize (bool): if True, loads full track and store normalization
            values based on the mixture file.
        ext (str): extension of audio files (default is .wav).
    """

    meta = {}
    path = Path(path)
    pendings = []
    # load test song names
    with open(TEST_SONG_LIST_FILE_PATH, "r") as fp:
        test_song_names = []
        for line in fp:
            line = line.strip()
            if line != "": test_song_names.append(line)
    #
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            song_name = Path(name).stem
            if song_name not in test_song_names:
                pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext)))
            # meta[name] = _track_metadata(root, sources, normalize, ext)
        for name, pending in tqdm.tqdm(pendings, ncols=120):
            meta[name] = pending.result()
    return meta



class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT, load_in_memory=False):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.

        Args:
            root (Path or str): root folder for the dataset.
            metadata (dict): output from `build_metadata`.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).

        samplerate and channels are converted on the fly.
        """
        self.load_in_memory = load_in_memory
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)
        assert len(self.num_examples)>0, f'No wavfile [{self.root}]'
        #
        if self.load_in_memory:
            self.tracks = self._preload()

    def _preload(self):
        tracks = dict()
        for name, meta in self.metadata.items():
            tracks[name] = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = torchaudio.load(str(file))
                wav = convert_audio_channels(wav, self.channels)
                tracks[name].append(wav)
        return tracks

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            if self.load_in_memory:
                wavs = [_[...,offset:num_frames] for _ in self.tracks[name]]
            else:
                for source in self.sources:
                    file = self.get_file(name, source)
                    wav, _ = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)
                    wav = convert_audio_channels(wav, self.channels)
                    wavs.append(wav)

            example = torch.stack(wavs)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example


####################################

Spec = namedtuple("Spec", "tempo onsets kr track index")


def rms(wav, window=10000):
    """efficient rms computed for each time step over a given window."""
    half = window // 2
    window = 2 * half + 1
    wav = F.pad(wav, (half, half))
    tot = wav.pow(2).cumsum(dim=-1)
    return ((tot[..., window - 1:] - tot[..., :-window + 1]) / window).sqrt()


def analyse_track(dset, index):
    """analyse track, extract bpm and distribution of notes from the bass line."""
    track = dset[index]
    mix = track.sum(0).mean(0)
    ref = mix.std()

    starts = (abs(mix) >= 1e-2 * ref).float().argmax().item()
    track = track[..., starts:]

    cache = CACHE / dset.sig
    cache.mkdir(exist_ok=True, parents=True)

    cache_file = cache / f"{index}.pkl"
    cached = None
    # if cache_file.exists():
    #     cached = try_load(cache_file)
    #     if cached is not None:
    #         tempo, events, hist_kr = cached

    if cached is None:
        drums = track[0].mean(0) # vocal percussion
        if drums.std() > 1e-2 * ref:
            tempo, events = beat_track(drums.numpy(), units='time', sr=SR)
        else:
            print("failed drums", drums.std(), ref)
            return None, track

        bass = track[1].mean(0) # bass
        r = rms(bass)
        peak = r.max()
        mask = r >= 0.05 * peak
        bass = bass[mask]
        if bass.std() > 1e-2 * ref:
            kr = torch.tensor(chroma_cqt(y=bass.numpy(), sr=SR))
            hist_kr = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
        else:
            print("failed bass", bass.std(), ref)
            return None, track

    pickle.dump([tempo, events, hist_kr], open(cache_file, 'wb'))
    spec = Spec(tempo, events, hist_kr, track, index)
    return spec, None


def best_pitch_shift(kr_a, kr_b):
    """find the best pitch shift between two chroma distributions."""
    deltas = []
    for p in range(12):
        deltas.append((kr_a - kr_b).abs().mean())
        kr_b = kr_b.roll(1, 0)

    ps = numpy.argmin(deltas)
    if ps > 6:
        ps = ps - 12
    return ps


def align_stems(stems):
    """Align the first beats of the stems.
    This is a naive implementation. A grid with a time definition 10ms is defined and
    each beat onset is represented as a gaussian over this grid.
    Then, we try each possible time shift to make two grids align the best.
    We repeat for all sources.
    """
    sources = len(stems)
    width = 5e-3  # grid of 10ms
    limit = 5
    std = 2
    x = torch.arange(-limit, limit + 1, 1).float()
    gauss = torch.exp(-x**2 / (2 * std**2))

    grids = []
    for wav, onsets in stems:
        le = wav.shape[-1]
        dur = le / SR
        grid = torch.zeros(int(le / width / SR))
        for onset in onsets:
            pos = int(onset / width)
            if onset >= dur - 1:
                continue
            if onset < 1:
                continue
            grid[pos - limit:pos + limit + 1] += gauss
        grids.append(grid)

    shifts = [0]
    for s in range(1, sources):
        max_shift = int(4 / width)
        dots = []
        for shift in range(-max_shift, max_shift):
            other = grids[s]
            ref = grids[0]
            if shift >= 0:
                other = other[shift:]
            else:
                ref = ref[shift:]
            le = min(len(other), len(ref))
            dots.append((ref[:le].dot(other[:le]), int(shift * width * SR)))

        _, shift = max(dots)
        shifts.append(-shift)

    outs = []
    new_zero = min(shifts)
    for (wav, _), shift in zip(stems, shifts):
        offset = shift - new_zero
        wav = F.pad(wav, (offset, 0))
        outs.append(wav)

    le = min(x.shape[-1] for x in outs)

    outs = [w[..., :le] for w in outs]
    return torch.stack(outs)


def find_candidate(spec_ref, catalog, pitch_match=True):
    """Given reference track, this finds a track in the catalog that
    is a potential match (pitch and tempo delta must be within the allowable limits).
    """
    candidates = list(catalog)
    random.shuffle(candidates)

    for spec in candidates:
        ok = False
        for scale in [1/4, 1/2, 1, 2, 4]:
            tempo = spec.tempo * scale
            delta_tempo = spec_ref.tempo / tempo - 1
            if abs(delta_tempo) < MAX_TEMPO:
                ok = True
                break
        if not ok:
            print(delta_tempo, spec_ref.tempo, spec.tempo, "FAILED TEMPO")
            # too much of a tempo difference
            continue
        spec = spec._replace(tempo=tempo)

        ps = 0
        if pitch_match:
            ps = best_pitch_shift(spec_ref.kr, spec.kr)
            if abs(ps) > MAX_PITCH:
                print("Failed pitch", ps)
                # too much pitch difference
                continue
        return spec, delta_tempo, ps


def get_part(spec, source, dt, dp):
    """Apply given delta of tempo and delta of pitch to a stem."""
    wav = spec.track[source]
    if dt or dp:
        wav = repitch(wav, dp, dt * 100, samplerate=SR, voice=source == 3)
        spec = spec._replace(onsets=spec.onsets / (1 + dt))
    return wav, spec


def build_track(ref_index, catalog):
    """Given the reference track index and a catalog of track, builds
    a completely new track. One of the source at random from the ref track will
    be kept and other sources will be drawn from the catalog.
    """
    order = list(range(len(SOURCES)))
    random.shuffle(order)

    stems = [None] * len(order)
    indexes = [None] * len(order)
    origs = [None] * len(order)
    dps = [None] * len(order)
    dts = [None] * len(order)

    first = order[0]
    spec_ref = catalog[ref_index]
    stems[first] = (spec_ref.track[first], spec_ref.onsets)
    indexes[first] = ref_index
    origs[first] = spec_ref.track[first]
    dps[first] = 0
    dts[first] = 0

    pitch_match = order != 0

    for src in order[1:]:
        spec, dt, dp = find_candidate(spec_ref, catalog, pitch_match=pitch_match)
        if not pitch_match:
            spec_ref = spec_ref._replace(kr=spec.kr)
        pitch_match = True
        dps[src] = dp
        dts[src] = dt
        wav, spec = get_part(spec, src, dt, dp)
        stems[src] = (wav, spec.onsets)
        indexes[src] = spec.index
        origs.append(spec.track[src])
    print("FINAL CHOICES", ref_index, indexes, dps, dts)
    stems = align_stems(stems)
    return stems, origs

def get_wav_dataset():
    root = Path(JACAPPELLA_PATH)
    ext = '.wav'
    metadata = build_metadata(root, SOURCES, ext=ext, normalize=False)
    train_set = Wavset(
        root, metadata, SOURCES, samplerate=SR, channels=CHANNELS,
        normalize=False, ext=ext)
    sig = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    train_set.sig = sig
    return train_set

def build_org_dataset():
    root = Path(JACAPPELLA_PATH)
    ext = '.wav'
    test_song_names = []
    with open(TEST_SONG_LIST_FILE_PATH, "r") as fp:
        test_song_names = []
        for line in fp:
            line = line.strip()
            if line != "": test_song_names.append(line)
    org_train_dir = ORG_OUTPATH / "train"
    org_test_dir = ORG_OUTPATH / "test"
    org_train_dir.mkdir(parents=True, exist_ok=True)
    org_test_dir.mkdir(parents=True, exist_ok=True)
    for filename in tqdm.tqdm(list(root.glob(f"**/*{ext}")), desc="Create org data"):
        target_dir = org_test_dir if filename.parent.stem in test_song_names else org_train_dir
        target_dir = target_dir / filename.parent.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(filename, target_dir / filename.name)

def main():
    random.seed(4321)
    if OUTPATH.exists():
        shutil.rmtree(OUTPATH)
    OUTPATH.mkdir(exist_ok=True, parents=True)
    (OUTPATH / 'train').mkdir(exist_ok=True, parents=True)
    (OUTPATH / 'valid').mkdir(exist_ok=True, parents=True)
    out = OUTPATH / 'train'

    build_org_dataset()

    dset = get_wav_dataset()
    dset2 = None
    dset3 = None
    pendings = []
    train_copies = 6
    valid_copies = 1 + 1

    with ProcessPoolExecutor(20) as pool:
        for index in range(len(dset)):
            pendings.append(pool.submit(analyse_track, dset, index))

        if dset2:
            for index in range(len(dset2)):
                pendings.append(pool.submit(analyse_track, dset2, index))
        if dset3:
            for index in range(len(dset3)):
                pendings.append(pool.submit(analyse_track, dset3, index))

        catalog = []
        for pending in tqdm.tqdm(pendings, ncols=120):
            spec, track = pending.result()
            if spec is not None:
                catalog.append(spec)
            else:
                raise ValueError

    for copy in range(train_copies):
        for index in range(len(catalog)):
            track, origs = build_track(index, catalog)
            mix = track.sum(0)
            mx = mix.abs().max()
            scale = max(1, 1.01 * mx)
            mix = mix / scale
            track = track / scale
            folder = out / f'{copy:02d}_{index:02d}'
            folder.mkdir()
            save_audio(mix, folder / "mixture.wav", SR)
            for stem, source, orig in zip(track, SOURCES, origs):
                save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                # save_audio(stem.std() * orig / (1e-6 + orig.std()), folder / f"{source}_orig.wav",
                #            SR, clip='clamp')

    out = OUTPATH / 'valid'
    for copy in range(valid_copies):
        for index in range(len(catalog)):
            if copy == 0:
                # copy original
                track = catalog[index].track
            else:
                track, origs = build_track(index, catalog)
            mix = track.sum(0)
            mx = mix.abs().max()
            scale = max(1, 1.01 * mx)
            mix = mix / scale
            track = track / scale
            folder = out / f'{copy:02d}_{index:02d}'
            folder.mkdir()
            save_audio(mix, folder / "mixture.wav", SR)
            for stem, source, orig in zip(track, SOURCES, origs):
                save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                # save_audio(stem.std() * orig / (1e-6 + orig.std()), folder / f"{source}_orig.wav",
                #            SR, clip='clamp')

if __name__ == '__main__':
    main()
