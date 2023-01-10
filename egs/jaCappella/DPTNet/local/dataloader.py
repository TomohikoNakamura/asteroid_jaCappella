from asteroid.data import jaCappellaCorpus
import torch
from pathlib import Path

def load_test_dataset(parser, args):
    test_dataset = jaCappellaCorpus(
        root=Path(args.test_dir),
        split="test",
        subset=None,
        sources=args.sources,
        targets=args.sources,
        segment=None,
        in_memory=True,
    )

    return test_dataset


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.train_dir),
    }

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = jaCappellaCorpus(
        split="train",
        sources=args.sources,
        targets=args.sources,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        in_memory=True,
        **dataset_kwargs,
    )

    valid_dataset = jaCappellaCorpus(
        split="valid",
        subset=None,
        sources=args.sources,
        targets=args.sources,
        segment=None,
        in_memory=True,
        **dataset_kwargs,
    )

    return train_dataset, valid_dataset

class Compose(object):
    """Composes several augmentation transforms.
    
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    gain = low + torch.rand(1) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio
