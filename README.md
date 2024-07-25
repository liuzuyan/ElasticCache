
# Efficient Inference of Vision Instruction-Following Models with Elastic Cache

This repository contains PyTorch implementation for Elastic Cache (ECCV 2024).

[Project Page](https://sites.google.com/view/elastic-cache) | [arXiv Paper](https://arxiv.org/)

## Elastic Cache

<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/07/19/8a94931d4958c0665a9bde8e35f1a974.png" alt="8a94931d4958c0665a9bde8e35f1a974.png" border="0">
</p>

Instruction encoding accounts for most of the theoretical computation cost, while the actual latency is negligible. This underscores that it’s not just model weights but also the **KV cache** used in output generation that can become a significant bottleneck. 

We propose **Elastic Cache** through a **Cache Merging** based on the importance scores of instruction tokens, complemented by a **fixed-point elimination** strategy in the output generation phase. Our designs yield significant inference acceleration while maintaining generation quality.

## Get Started

1. **Environmental Setup**: 

    We choose LLaVA-1.5 and Qwen-VL as our base model. You can install following dependencies for Elastic Cache evaluation:

   ```
   pip install -r requirements.txt
   ```
   
2. **Initial Weights**: 

    We use [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b), [LLaVA-1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b) and [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) in our experiements, you may download these models and put them at /path/to/model

3. **Download Eval Data**: 

   You can download our pre-processed MM-Vet dataset [here](https://drive.google.com/file/d/1MLB7Pr_zo2Nu5iihuXRXE38nHzY-TnRN/view?usp=sharing), and put it at `./playground/data/mm-vet`. Our choosed LLaVA-Description datasets will come soon. 
   
   You can also prepare your own conversations for testing following the format in the json file. 

4. **Eval** 

    Please refer to [EVAL.md](https://github.com/liuzuyan/ElasticCache/blob/main/EVAL.md) for the detailed instructions on evaluation, including generation, PPL evaluation, ROUGE evaluation, and latency test. 

## Quantitative and Qualitative Results

We evaluate **Elastic Cache** together with baselines (H2O and StreamingLLM) on PPL (lower better) and ROUGE (higher better) metrics. We conduct LLaVA-1.5 of different sizes (a),(b) and Qwen-VL-7B(c) for visual tasks. Our Elastic Cache outperforms baselines consistently.

<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/07/19/30dcc0713f9c3dc40600846aa2037509.png" alt="30dcc0713f9c3dc40600846aa2037509.png" border="0">
</p>

## Citation

If you found this repository useful, please consider citing:

``` 
@article{liu2024elastic,
title={Efficient Inference of Vision Instruction-Following Models with Elastic Cache},
author={Liu, Zuyan and Liu, Benlin and Wang, Jiahui and Dong, Yuhao and Chen, Guangyi and Rao, Yongming and Krishna, Ranjay and Lu, Jiwen},
journal={arXiv preprint arXiv:},
year={2024}
}
```