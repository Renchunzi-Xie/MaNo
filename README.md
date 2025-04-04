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
We conduct large-scale experiments under several distribution shifts with ResNets, ConvNext, and ViT. 

🥇 **SOTA performance.** Our approach **MaNo** provides the best and most robust accuracy estimation.

🚀 **Qualitative benefits.** Our approach is training-free, fast and memory efficient.
<p align="center">
<img src="https://github.com/user-attachments/assets/b2baa7d4-06b6-4435-9ffc-3b730e9bc76e" height="250"> &nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/94f84f43-eabe-4d0c-9557-6a22063d2759" height="250">
</p>

## Datasets
### Pre-trained process

1. CIFAR10 & CIFAR-100 can be downloaded in the code. 
2. Download TinyImageNet
```angular2html
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
3. Download ImageNet
```angular2html
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
```
4. Download Office-Home
```angular2html
pip install gdown==4.6.0
gdown https://drive.google.com/uc?id=1JMFEHM46xmgp2RSX6iVgcR5fCpZkeruJ
```
5. Download PACS
```angular2html
https://www.kaggle.com/datasets/nickfratto/pacs-dataset
```
6. Download DomainNet
```angular2html
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
```
### Evaluation process
1. Download CIFAR-10C & CIFAR-100C
```angular2html
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
```
2. Download TinyImageNet-C
```angular2html
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
3. Download ImageNet-C
```angular2html
wget https://zenodo.org/record/2235448/files/blur.tar
wget https://zenodo.org/record/2235448/files/digital.tar
wget https://zenodo.org/record/2235448/files/extra.tar
wget https://zenodo.org/record/2235448/files/noise.tar
wget https://zenodo.org/record/2235448/files/weather.tar
```

## Usage
Step 1: Pre-train models on CIFAR-10, CIFAR-100 and TinyImageNet using commands in `./bash/init_base_model.sh`.

Step 2: Estimate OOD error on CIFAR-10C, CIFAR-100C, and TinyImageNet-C using commands in `./bash/mano.sh`.

You can simply use the 'main' branch to reproduce the results. 

## Authors
- [Renchunzi Xie](https://scholar.google.com/citations?user=EQSNE-wAAAAJ&hl=zh-CN)
- [Ambroise Odonnat](https://ambroiseodt.github.io/)
- [Vasilii Feofanov](https://vfeofanov.github.io/)

## Licence
The code is distributed under the MIT license.

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{xie2024manoexploitingmatrixnorm,
 author = {Xie, Renchunzi and Odonnat, Ambroise and Feofanov, Vasilii and Deng, Weijian and Zhang, Jianfeng and An, Bo},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {41767--41802},
 publisher = {Curran Associates, Inc.},
 title = {MaNo: Exploiting Matrix Norm for Unsupervised Accuracy Estimation Under Distribution Shifts},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/49abf767d606b72f74ea6009176fafeb-Paper-Conference.pdf},
 volume = {37},
}
```
