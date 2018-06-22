# image-segmentation-tensorflow
Lots of Image Segmentation implements with Tensorflow

## Prerequisites
* python 3.x
* tensorflow 1.x
* numpy
* scipy (some features are about to **deprecated**, they'll be replaced)
* scikit-image
* opencv-python
* pillow
* h5py
* tqdm
* Internet :)

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
* FC-DenseNet
* E/Link-Net
* RefineNet
* PSPNet
* Mask-RCNN
* DecoupledNet
* GAN-SS

## DataSets
Now supporting(?) DataSets are... (code is in /datasets.py)
* MSCOCO

## Repo Tree
```
│
├── xxNet
│    ├──gan_img (generated images)
│    │     ├── train_xxx.png
│    │     └── train_xxx.png
│    ├── model  (model)
│    │     ├── checkpoint
│    │     ├── ...
│    │     └── xxx.ckpt
│    ├── xxx_model.py (model)
│    ├── xxx_train.py (trainer)
│    ├── xxx_tb.png   (Tensor-Board result)
│    └── readme.md    (results & explains)
├── tfutil.py         (useful TF utils)
├── image_utils.py    (image processing)
└── datasets.py       (DataSet loader)
```

## To-Do
1. Implement FCN
2. Implement U-NET
3. -

## ETC

**Any suggestions and PRs and issues are WELCONE :)**

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
