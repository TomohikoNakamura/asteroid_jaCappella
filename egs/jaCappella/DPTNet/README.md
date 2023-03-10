# DPTNet for vocal ensemble separation

# Setup
- Setup the conda environment of asteroid (see asteroid web page)
    - If the installation hangs, execute `conda config --set channel_priority strict`.
- Install some packages
```
pip install -r requirements.txt
```

# Separating signals
- Use the pretrained model
```
python separate.py --model_dir pretrained --output_dir /path/to/output/dir /path/to/to-be-separated/mixture/file
```

# Train your own model
## Setup
- Modify `/path/to/your/python/site-packages/pb_bss_eval/evaluation/wrapper.py` becuase `pb_bss_eval` only supports up to five sources.
    - line 359: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
    - line 364: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
- Once the above modification is done, there is no need to do it again.
- Create augmented dataset (see [here](../README.md#how-to-create-augmented-data))

## Training model
```
python train.py --train_dir /path/to/augmented/jaCappella/data --output /path/to/output/dir
```

## Evaluating model
```
python eval.py --test_dir /path/to/jaCappella/data --model_dir /path/to/output/dir
```

# References
- Saurjya Sarkar, Emmanouil Benetos, and Mark Sandler, "Vocal harmony separation using time-domain neural networks," in Proceedings of INTERSPEECH, 2021, pp. 3515--3519.

