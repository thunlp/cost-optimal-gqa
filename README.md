<h1 align="center">Cost-Optimal Grouped-Query Attention for Long-Context Modeling</h1>

<div align="center">
  <a href="https://huggingface.co/collections/chen-yingfa/cost-optimal-gqa-models-68c685bab808768393c9aa39">🤗 Models</a> |
  <a href="https://arxiv.org/abs/2310.05963">Paper (arXiv)</a>
</div>
</br>

<div align="center">
  <span style="font-weight: bold;">Yingfa Chen*, Yutong Wu*, Chenyang Song, Zhen Leng Thai, Xingyu Shen, Xu Han, Zhiyuan Liu, Maosong Sun</span> </br>
  Tsinghua University, University of Science and Technology Beijing</br>
  <span style="font-family: monospace">chenyingfa1999@gmail.com, wuyutong_yuna@163.com</span> </br></br>
</div>

This repository contains the code and models used in the paper [Cost-Optimal Grouped-Query Attention for Long-Context Modeling](https://arxiv.org/abs/2503.09579).

## How to Run the Code

### Step 1: Setup

First, setup the environment and download the pretraining dataset by following the README in the `src` folder.

### Step 2: Train the Models

In the `src` folder, execute the following. 

```shell
bash train.sh
```

This will automatically use multiple GPUs is available. To replicate the numbers in the paper, please run the training experiments on machines with 8 NVIDIA A800-80GB GPUs.
