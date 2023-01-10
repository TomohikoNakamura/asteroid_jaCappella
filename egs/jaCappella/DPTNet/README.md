# DPTNet for vocal ensemble separation

# Setup
- Setup the conda environment of asteroid (see asteroid web page)
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
## Training model
```
python train.py --train_dir /path/to/augmented/jaCappella/data --output /path/to/output/dir
```

## Evaluating model
- Before evaluation, modify `/path/to/your/python/site-packages/pb_bss_eval/evaluation/wrapper.py` becuase the evaluation library only supports up to five sources.
    - line 359: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
    - line 364: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
- Once the above evaluation is done, there is no need to modify it again.
- Execute `eval.py`
```
python eval.py --test_dir /path/to/jaCappella/data --model_dir /path/to/output/dir
```

# References
- Saurjya Sarkar, Emmanouil Benetos, and Mark Sandler, "Vocal harmony separation using time-domain neural networks," in Proceedings of INTERSPEECH, 2021, pp. 3515--3519.

