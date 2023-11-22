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
DiffSeg is an unsupervised zero-shot segmentation method using attention information from a stable-diffusion model. This repo implements the main DiffSeg algorithm and addtionally include an experimental feature to add semantic labels to the masks based on a generated caption.

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

## Jupyter Notebook
Please see the instructions in the ``diffseg.ipynb`` for running instructions. 


