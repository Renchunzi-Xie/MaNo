## MaNo (NeurIPS'24)
**The repository contains the official implementation of MaNo, a training-free method to assess the generalization performance of neural networks under distribution shifts introduced in** 

>[MANO: Exploiting Matrix Norm for Unsupervised Accuracy Estimation Under Distribution Shifts](https://arxiv.org/pdf/2405.18979).
><br/>Renchunzi Xie*, Ambroise Odonnat*, Vasilii Feofanov*, Weijian Deng, Jianfeng Zhang, Bo An.
<br/>*Equal contribution.

## Overview
**MaNo** is an efficient training-free approach grounded in theory that leverages logits to estimate the generalization performance of a pre-trained neural network under distribution shifts. It makes use of $\sigma$, a novel normalization function to deal with poorly-calibrated scenarios. Given a *pre-trained* model $f$ and a test set $D_{test} = (x_i)_{i=1}^N$ with K classes, **MaNo** operates as follows:
- Recover the logits $\theta_i = f(x_i)$
- Normalize the logits $\theta_i \to \sigma(\theta_i) \in \Delta_K$
- Fill a prediction matrix $P \in \mathbb{R}^{N \times K}$ with rows $\sigma(\theta_i)$
- Compute the estimation score $S(f, D_{test})$ as the scaled p-norm of $P$, i.e., $S(f, D_{test}) = \frac{1}{NK} \lVert P \rVert_p$.

## Results
We conduct large-scale experiments under several distribution shifts with ResNets, ConvNext, and ViT on various open-source benchmarks. 

🥇 **SOTA performance.** Our approach **MaNo** provides the best and most robust accuracy estimation.

🚀 **Qualitative benefits.** Our approach is training-free, fast and memory efficient.
<p align="center">
<img src="https://github.com/user-attachments/assets/b2baa7d4-06b6-4435-9ffc-3b730e9bc76e" height="250"> &nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/94f84f43-eabe-4d0c-9557-6a22063d2759" height="250">
</p>

## Installation

## Modules

## Usage

## Open-source Participation
Do not hesitate to contribute to this project, we would be happy to receive feedback and integrate your suggestions.

## Authors
- [Renchunzi Xie](https://scholar.google.com/citations?user=EQSNE-wAAAAJ&hl=zh-CN)
- [Ambroise Odonnat](https://ambroiseodt.github.io/)
- [Vasilii Feofanov](https://vfeofanov.github.io/)

## Licence
The code is distributed under the MIT license.

## Citation
If you find this work useful in your research, please cite:
```
@misc{xie2024manoexploitingmatrixnorm,
      title={MANO: Exploiting Matrix Norm for Unsupervised Accuracy Estimation Under Distribution Shifts}, 
      author={Renchunzi Xie and Ambroise Odonnat and Vasilii Feofanov and Weijian Deng and Jianfeng Zhang and Bo An},
      year={2024},
      eprint={2405.18979},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.18979}, 
}
```
