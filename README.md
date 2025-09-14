# Cost-Optimal GQA for Long-Context Modeling

This repository contains the code and models used in the pape [Cost-Optimal Grouped-Query Attention for Long-Context Modeling](https://arxiv.org/abs/2503.09579).

## How to Run the Code

### Step 1: Setup

First, setup the environment and download the pretraining dataset by following the README in the `src` folder.

### Step 2: Train the Models

In the `src` folder, execute the following. 

```shell
bash train.sh
```

This will automatically use multiple GPUs is available. To replicate the numbers in the paper, please run the training experiments on machines with 8 NVIDIA A800-80GB GPUs.
