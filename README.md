# Detection Module Zoo 

It is a Module Zoo for object detection based on two-stage algorithm (Faster R-CNN). For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. And the basic framework of Faster R-CNN are implemented in pure python3 environment. It have been tested on cpu-only, nvi-gpu and Mac-os machine (without dependency on CUDA-CUDNN-NVCC).



## Requirements: software

``Tensorflow ``

``cython``, ``python-opencv``, ``easydict``



## Requirements: hardware

For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

## Installation 

1. Clone the Faster R-CNN repository

   ```bash
   # Make sure to clone with --recursive
   git clone --recursive https://github.com/smallcorgi/Faster-RCNN_TF.git
   ```

   

2. Build the Cython modules

   ```bash
   cd $FRCN_ROOT/lib
   make
   ```

   

## Demo

*After successfully completing basic installation*, you'll be ready to run the demo.

Download model training on PASCAL VOC 2007 [[Google Drive\]](https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM) [[Dropbox\]](https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt?dl=0)

To run the demo

```bash
cd $FRCN_ROOT
python ./tools/demo.py --model model_path
```





## Training Model



1. Download the training, validation, test data and VOCdevkit

   ```bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   Extract all of th
   ```

   

2. Extract all of these tars into one directory named `VOCdevkit`

   ```bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

   

3. It should have this basic structure

   ```bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

   

4. Create symlinks for the PASCAL VOC dataset

   ```bash
   cd $FRCN_ROOT/data
   ln -s $VOCdevkit VOCdevkit2007
   ```

   

5. Download pre-trained ImageNet models
   [[Google Drive\]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox\]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)

   ```bash
   mv VGG_imagenet.npy $FRCN_ROOT/data/pretrain_model/VGG_imagenet.npy
   ```

   

6. Run script to train and test model

   ```bash
   cd $FRCN_ROOT
   ./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 pascal_voc
   ```

   



