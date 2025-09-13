# Cost-Optimal GQA for Long-Context Modeling

This repository contains the code and models used in the pape [Cost-Optimal Grouped-Query Attention for Long-Context Modeling](https://arxiv.org/abs/2503.09579).

## How to Run the Code

Execute the following. This will automatically use multiple GPUs is available. To replicate the numbers in the paper, please run the training experiments on machines with 8 NVIDIA A800-80GB GPUs.
```shell
bash train.sh
```
