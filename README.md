# Diffuse, Attend, and Segment
 This repo implements the DiffSeg segmentation method in the paper [``Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion``](https://arxiv.org/abs/2308.12469).
 
 ```
 @article{tian2023diffuse,
  title={Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion},
  author={Tian, Junjiao and Aggarwal, Lavisha and Colaco, Andrea and Kira, Zsolt and Gonzalez-Franco, Mar},
  journal={arXiv preprint arXiv:2308.12469},
  year={2023}
}
```

## Overview
DiffSeg is an unsupervised zero-shot segmentation method using attention information from a stable-diffusion model. This repo implements the main DiffSeg algorithm and additionally include an experimental feature to add semantic labels to the masks based on a generated caption.

More details can be found on the project page: https://sites.google.com/corp/view/diffseg/home

## Create conda environment

- The environment uses Ubuntu 18.04 and Tensorflow 2.14 supported on CUDA 11.x and python 3.9. 
```
cd diffseg
conda create --name diffseg python=3.9
conda activate diffseg
pip install -r path/to/requirements.txt
```

## Computation Requirement
- We recommend using 2 GPUs with a minimum 11G VRAM each, e.g., RTX2080Ti.
- One GPU is for loading the Stable Diffusion model and the other is for the BLIP captioning model. 

## DiffSeg Notebook
Please see the instructions in the ``diffseg.ipynb`` for running instructions. 

## Benchmarks
We benchmark the performance of DiffSeg on [CoCo-Stuff-27](https://github.com/nightrome/cocostuff) and [Cityscapes](https://www.cityscapes-dataset.com/). Please see instructions in ``benchmarks.ipynb``.
* We follow the evaluation protocol in [PiCIE](https://sites.google.com/view/picie-cvpr2021/home) and use the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) for matching predictions and ground truth labels. 

## Contributors
- **Junjiao Tian (Google and Georgia Tech)**
- **Lavisha Aggarwal (Google)**
- **Andrea Colaco (Google)**
- **Zsolt Kira (Georgia Tech)**
- **Mar Gonzalez-Franco (Google)**  
