<div align="center">
<h1> Revisiting Neural Networks for Continual Learning:<br />
  An Architectural Perspective
</h1>
<div>
    <a>Aojun Lu</a>&emsp;
    <a target='_blank'>Tao Feng</a>&emsp;
    <a href='https://jacobyuan7.github.io/' target='_blank'>Hangjie Yuan</a>&emsp;
    <a>Xiaotian Song</a>&emsp;
    <a href='https://yn-sun.github.io/' target='_blank'>Yanan Sun&#9993</a>&emsp;
</div>

<strong>Accepted to <a href='https://ijcai24.org/' target='_blank'>IJCAI 2024</a> :partying_face:</strong>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.07965)
[![GitHub Stars](https://img.shields.io/github/stars/byyx666/ArchCraft?style=social)](https://github.com/byyx666/ArchCraft)
[![GitHub Forks](https://img.shields.io/github/forks/byyx666/ArchCraft)](https://github.com/byyx666/ArchCraft)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbyyx666%2FArchCraft&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</div>

> Abstract:
> Efforts to overcome catastrophic forgetting have primarily centered around developing more effective Continual Learning (CL) methods. In contrast, less attention was devoted to analyzing the role of network architecture design (e.g., network depth, width, and components) in contributing to CL. This paper seeks to bridge this gap between network architecture design and CL, and to present a holistic study on the impact of network architectures on CL. This work considers architecture design at the network scaling level, i.e., width and depth, and also at the network components, i.e., skip connections, global pooling layers, and down-sampling. In both cases, we first derive insights through systematically exploring how architectural designs affect CL. Then, grounded in these insights, we craft a specialized search space for CL and further propose a simple yet effective ArchCraft method to steer a CL-friendly architecture, namely, this method recrafts AlexNet/ResNet into AlexAC/ResAC. Experimental validation across various CL settings and scenarios demonstrates that improved architectures are parameter-efficient, achieving state-of-the-art performance of CL while being 86%, 61%, and 97% more compact in terms of parameters than the naive CL architecture in Task IL and Class IL. 

## Information before using this repo
I changed all the paths to prevent possible information leakage.
In order to run the code, you will need to configure the paths to match your own system (see class_il/utils/data.py and task_il/dataloaders).

⭐⭐⭐Consider starring the repo! ⭐⭐⭐

## How to Use
**First, select a continual learning scenario you are interested in. And then:**

**To evaluate the performance of a given architecture:**

Run test.py

**To design a new architecture using ArchCraft:**

Make sure the value of 'is_running' in global.ini is 0, unless you want to continue the previous search process that was interrupted. 
Then run evolve.py

## Citation
If you find this repo useful, please consider citing our paper.
```bibtex
@article{lu2024revisiting,
  title={Revisiting Neural Networks for Continual Learning: An Architectural Perspective},
  author={Lu, Aojun and Feng, Tao and Yuan, Hangjie and Song, Xiaotian and Sun, Yanan},
  journal={arXiv preprint arXiv:2404.14829},
  year={2024}
}
```

## Acknowledgement
Part of this work's implementation refers to several prior works including [CNN-GA](https://github.com/yn-sun/cnn-ga), [PyCIL](https://github.com/G-U-N/PyCIL), and [HAT](https://github.com/joansj/hat).
