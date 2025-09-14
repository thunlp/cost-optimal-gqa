# Training Code for Reproducing Cost-Optimal GQA

An simple but fast implementation of pre-training language models, implemented with PyTorch + 🤗 Accelerate.

## Installation

This requires:

- Python 3.10+ (tested on 3.12)
- PyTorch 2.0+

Run `pip install -r requirements.txt` to install the dependencies.

## Data

Please download SlimPajama from: https://huggingface.co/datasets/AlppAI/SlimPajama-chunked

This dataset is the same as the original data, but multiple files are merged for faster downloading and processing.

### How to Use Other Datasets

See the README file in `data/` for more details.

## How to Run?

The easiest way is to simply run:

```shell
bash train.sh train_config=60m_8k model_config=gpt-4-512 data_path=/path/to/data
```

By default, this will train the smallest model (3M parameters on 60M tokens) in our scaling law experiments (the Step 2 of our method in the paper).

### Passing Arguments

`train.sh` can accept arguments (`<args>`) and pass them to `train.py`. For instance, if I want to use Grouped Query Attention instead of Multi-Head Attention (the default), I can run:

```shell
bash train.sh n_head=32 n_kv_head=8
```

> Note that you must use `=` in the arguments after `train.sh`.

### Configuration

The configuration files are in `configs/training/` and `configs/model/`. You can pass in `--train_config <path_to_config_file>` and `--model_config <path_to_config_file>` to specify which configuration to use. Additional arguments can be specified in the command line, and they will override the values in the configuration file. See `arguments.py` for all the arguments.

## Acknowledgements

Much of the code comes from:

- https://github.com/karpathy/nanoGPT
- https://github.com/whyNLP/tinyllama

Also, big thanks to the HuggingFace and PyTorch team for their awesome tools.
