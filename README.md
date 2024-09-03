This repo is the PyTorch implementation of our paper:

**["Rethinking Barely-Supervised Volumetric Medical Image Segmentation from an Unsupervised Domain Adaptation Perspective"](https://arxiv.org/abs/2405.09777)** 

<!-- <img src=docs/BvA.png width=75% /> -->

**B**arely-supervised learning **v**ia unsupervised domain **A**daptation (BvA) 

## Usage

### 0. Requirements
The code is developed using Python 3.8 with PyTorch 1.11.0. 
All experiments in our paper were conducted on a single NVIDIA Quadro RTX 6000 with 24G GPU memory.

### 1. Data Preparation
#### 1.1. Download data
The original data can be download in the following link:
* [LA Dataset](https://www.cardiacatlas.org/atriaseg2018-challenge/)

The preprecessed data can be downloaded from UA-MT:
* [LA Dataset](https://github.com/yulequan/UA-MT/tree/master/data)

#### 1.2. Split Dataset
The LA dataset contains 100 gadolinium-enhanced MRI scans. We split LA into 80 samples for training (where we further divided 80 samples into 70 for training and 10 for validation) and 20 samples for testing.
The splited lists of samples are as follows:
```
exp_la/data/
|-- train.list
|-- val.list
|-- test.list
```

### 2. Training
```angular2html
python train_bva.py
```

### 3. Evaluation
```angular2html
python eval.py
```


## Citation
If you find this project useful, please consider citing:
```
@article{shen2024rethinking,
  title={Rethinking Barely-Supervised Segmentation from an Unsupervised Domain Adaptation Perspective},
  author={Shen, Zhiqiang and Cao, Peng and Su, Junming and Yang, Jinzhu and Zaiane, Osmar R},
  journal={arXiv preprint arXiv:2405.09777},
  year={2024}
}
```

## Contact
If you have any questions or suggestions, please feel free to contact me ([xxszqyy@gmail.com](xxszqyy@gmail.com)).
