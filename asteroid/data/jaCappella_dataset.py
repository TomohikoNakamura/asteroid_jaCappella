from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf
from collections import OrderedDict

class jaCappellaCorpus(torch.utils.data.Dataset):
    """jaCappella corpus: Japanese a cappella vocal ensemble corpus

    The corpus consists of 35 copyright-cleared vocal ensemble songs.
    Separate audio recordings of six voice parts (`lead_vocal`, `soprano`, `alto`, `tenor`, `bass`, and `vocal_percussion`) are available.

    - jaCappella corpus website: https://tomohikonakamura.github.io/jaCappella_corpus/

    .. note::
        The corpus is downloadable via the above website and require that users agree all the statements of the terms of use of the corpus.

    This implementation is based on the implementation of the MUSDB18Dataset class.

    Args:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names 
            that composes the mixture.
            (Defaults: `lead_vocal`, `soprano`, `alto`, `tenor`, `bass`, `vocal_percussion`)
        targets (list or None, optional): List of source names to be used as
            targets. If None, a dict with the 4 stems is returned.
             If e.g [`soprano`, `bass`], a tensor with stacked `soprano` and
             `bass` is returned instead of a dict. Defaults to None.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: Enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        in_memory (boolean, optional): Enables preloading of all audio files.

    Attributes:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names 
            that composes the mixture.
            (Defaults: `lead_vocal`, `soprano`, `alto`, `tenor`, `bass`, `vocal_percussion`)
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        tracks (:obj:`list` of :obj:`Dict`): List of track metadata
        in_memory (boolean, optional): Enables preloading of all audio files.

    References
        T. Nakamura, S. Takamichi, N. Tanji, S. Fukayama, and H. Saruwatari, "jaCappella corpus: A Japanese a cappella vocal ensemble corpus," arXiv preprint: 2211.16028, 2022.
    """

    dataset_name = "jaCappella"
    MIXTURE = "mixture"

    def __init__(
        self,
        root,
        sources=["vocal_percussion", "bass", "alto", "tenor", "soprano", "lead_vocal"],
        targets=None,
        suffix=".wav",
        split="train",
        subset=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=48000,
        in_memory=False,
    ):
        self.in_memory= in_memory
        if self.in_memory:
            print("Data will be preloaded in memory")
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        if self.random_track_mix and self.split != "train":
            raise ValueError('Random track mix can be used only for training.')
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def get_track_path(self, index):
        return self.tracks[index]["path"]

    def __getitem__(self, index):
        # assemble the mixture of target and interferers
        audio_sources = OrderedDict()

        # get track_id
        track_id = index // self.samples_per_track
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # load sources
        for source in self.sources:
            # optionally select a random track for each source
            if self.random_track_mix:
                # load a different track
                track_id = random.choice(range(len(self.tracks)))
                if self.random_segments:
                    start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

            # loads the full track duration
            start_sample = int(start * self.sample_rate)
            # check if dur is none
            if self.segment:
                # stop in soundfile is calc in samples, not seconds
                stop_sample = start_sample + int(round(self.segment * self.sample_rate))
            else:
                # set to None for reading complete file
                stop_sample = None

            # load actual audio
            if self.in_memory:
                audio = self.tracks[track_id][source][start_sample:stop_sample,:] # time x ch
            else:
                audio, _ = sf.read(
                    Path(self.tracks[track_id]["path"] / source).with_suffix(self.suffix),
                    always_2d=True,
                    start=start_sample,
                    stop=stop_sample,
                ) # time x ch
            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float) # ch x time
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)
            audio_sources[source] = audio

        # apply linear mix over source index=0
        if self.split == "train":
            audio_mix = torch.stack(list(audio_sources.values())).sum(0)
        else:
            if self.in_memory:
                audio_mix = self.tracks[track_id][self.MIXTURE][start_sample:stop_sample,:] # time x ch
            else:
                audio_mix, _ = sf.read(
                    Path(self.tracks[track_id]["path"] / self.MIXTURE).with_suffix(self.suffix),
                    always_2d=True,
                    start=start_sample,
                    stop=stop_sample,
                ) # time x ch
            audio_mix = torch.tensor(audio_mix.T, dtype=torch.float) # ch x time
        if self.targets:
            audio_sources = torch.stack(
                [wav for src, wav in audio_sources.items() if src in self.targets], dim=0
            )
        return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(list(p.iterdir()), desc=f"Loading {self.split} data"):
            if track_path.is_dir():
                if self.subset and track_path.stem not in self.subset:
                    # skip this track
                    continue

                source_paths = [track_path / (s + self.suffix) for s in self.sources]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track due to non-existing source", track_path)
                    continue

                # get metadata
                infos = list(map(sf.info, source_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to different sample rate ", track_path)
                    continue

                if self.segment is not None:
                    # get minimum duration of track
                    min_duration = min(i.duration for i in infos)
                    if min_duration > self.segment:
                        if self.in_memory:
                            data = {"path": track_path, "min_duration": min_duration}
                            for source in self.sources:
                                signal, _ = sf.read(
                                    Path(track_path / source).with_suffix(self.suffix),
                                    always_2d=True
                                )
                                data[source] = signal
                            yield (data)
                        else:
                            yield ({"path": track_path, "min_duration": min_duration})
                else:
                    if self.in_memory:
                        data = {"path": track_path, "min_duration": None}
                        for source in self.sources:
                            signal, _ = sf.read(
                                Path(track_path / source).with_suffix(self.suffix),
                                always_2d=True
                            )
                            data[source] = signal
                        if self.split != "train":
                            signal, _ = sf.read(Path(track_path / self.MIXTURE).with_suffix(self.suffix), always_2d=True)
                        data[self.MIXTURE] = signal
                        yield (data)
                    else:
                        yield ({"path": track_path, "min_duration": None})

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [jaCappella_license]
        return infos


jaCappella_license = dict(
    title="jaCappella corpus",
    title_link="https://tomohikonakamura.github.io/jaCappella_corpus/",
    author="Tomohiko Nakamura, Shinnosuke Takamichi, Naoko Tanji, and Hiroshi Saruwatari",
    license="jaCappella License",
    license_link="https://tomohikonakamura.github.io/jaCappella_corpus/",
)
