# CrossNet-Open-Unmix (X-UMX)

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
## Setup
- Modify `/path/to/your/python/site-packages/pb_bss_eval/evaluation/wrapper.py` becuase `pb_bss_eval` only supports up to five sources.
    - line 359: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
    - line 364: `assert self.K_source <= 5, _get_err_msg(` -> `assert self.K_source <= 10, _get_err_msg(`
- Once the above modification is done, there is no need to do it again.

## Training model
```
python train.py --train_dir /path/to/augmented/jaCappella/data --output /path/to/output/dir
```

## Evaluating model
```
python eval.py --test_dir /path/to/jaCappella/data --model_dir /path/to/output/dir
```

# References
- Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi, and Yuki Mitsufuji, "All for One and One for All: Improving Music Separation by Bridging Networks," arxiv 2010.04228, 2021.

