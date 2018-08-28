# Awesome-Segmentation
Lots of Image Semantic Segmentation Implementations in **Tensorflow/Keras**

**Highly inspired by [HERE](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)**

## Prerequisites
* python 3.x
* tensorflow 1.x
* keras 2.x
* numpy
* scikit-image
* opencv-python
* h5py
* tqdm

## Usage
### Dependency Install
    $ sudo python3 -m pip install -r requirements.txt
### Training Model
    (Before running train.py, MAKE SURE run after downloading DataSet & changing DataSet's directory in xxx_train.py)
    just after it, RUN train.py
    $ python3 xxx_train.py

## Implementation List
* FCNet
* SegNet
* U-Net
* FusionNet
* FC-DenseNet
* ENet
* LinkNet
* RefineNet
* PSPNet
* Mask R-CNN
* DecoupledNet
* GAN-SS
* G-FRNet

## DataSets
* ``MS COCO 2017`` DataSet will be used!

*DataSet* | *Train* | *Validate* | *Test* | *Disk*
:---: | :---: | :---: | :---: | :---: |
**MS COCO 2017** | 118287 | 5000 | 40670 | ``26.3GB``

## Repo Tree
```
│
├── xxNet
│    ├──gan_img (generated images)
│    │     ├── train_xxx.png
│    │     └── train_xxx.png
│    ├── model  (model)
│    │     └── model.txt (google-drive link for pre-trained model)
│    ├── xxx_model.py (model)
│    ├── xxx_train.py (trainer)
│    ├── xxx_tb.png   (Tensor-Board result)
│    └── readme.md    (results & explains)
├── tfutil.py         (useful TF utils)
├── image_utils.py    (image processing)
└── datasets.py       (DataSet loader)
```

## Pre-Trained Models

Here's a **google drive link**. You can download pre-trained models from [~~here~~]() !

## Papers & Codes

*Name* | *Summary* | *Paper* | *Code*
:---: | :---: | :---: | :---:
**FCN** | *Fully Convolutional Networks for Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1411.4038) |
**SegNet** | *A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation* | [[arXiv]](https://arxiv.org/abs/1511.00561) |
**U-Net** | *Convolutional Networks for Biomedical Image Segmentation* | [[arXiv]](https://arxiv.org/abs/1505.04597) | 
**FusionNet** | *A deep fully residual convolutional neural network for image segmentation in connectomics* | [[arXiv]](https://arxiv.org/abs/1612.05360) |
**FC-DenseNet** | *Fully Convolutional DenseNets for Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1611.09326) |
**ENet** | *A Deep Neural Network Architecture for Real-Time Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1606.02147) |
**LinkNet** | *Exploiting Encoder Representations for Efficient Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1707.03718) |
**Mask R-CNN** | *Mask R-CNN* | [[arXiv]](https://arxiv.org/abs/1703.06870) |
**PSPNet** | *Pyramid Scene Parsing Network* | [[arXiv]](https://arxiv.org/abs/1612.01105) |
**RefineNet** | *Multi-Path Refinement Networks for High-Resolution Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1611.06612) |
**G-FRNet** | *Gated Feedback Refinement Network for Dense Image Labeling* | [[CVPR2017]](http://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf) |
**DeepLabv3+** | *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* | [[arXiv]](https://arxiv.org/abs/1802.02611) |
**DecoupledNet** | *Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation* | [[arXiv]](https://arxiv.org/abs/1506.04924) |
**GAN-SS** | *Semi and Weakly Supervised Semantic Segmentation Using Generative Adversarial Network* | [[arXiv]](https://arxiv.org/abs/1703.09695) |

## To-Do
1. Implement FCN
2. Implement Mask R-CNN
3. Upload U-Net (Tuned)
4. Upload FC-DenseNet
5. Upload DeepLabv3+

## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
