# Detection Module Zoo 

It is a Module Zoo for object detection based on two-stage algorithm (Faster R-CNN). For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. And the basic framework of Faster R-CNN are implemented in pure python3 environment (mainly inspired by [endernewton's implement](https://github.com/endernewton/tf-faster-rcnn)). It have been tested on cpu-only, nvi-gpu and Mac-os machine (without dependency on CUDA-CUDNN-NVCC).



## Requirements: software

``Tensorflow ``

``cython``, ``python-opencv``, ``easydict``



## Requirements: hardware

For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

## Installation 

1. Clone the Faster R-CNN repository

   ```bash
   # Make sure to clone with --recursive
   git clone --recursive https://github.com/KerrWu/detection_block_zoo.git
   ```

   

2. Build the Cython modules

   To make at cpu-only machine (or other machine without CUDA, i.e, mac), I comment some code related to NVCC. Basically just some NMS code which use GPU to accelerate.
   
   
   
```bash
   cd $FRCN_ROOT/lib
   make clean
   make
   cd ..
   
   
   cd data
   git clone https://github.com/pdollar/coco.git
   cd coco/PythonAPI
   make
   cd ../../..
```

   

   

   

   

## Demo

After successfully completing basic installation, you'll be ready to run the demo.

Please download pre-train model on PASCAL VOC 2007 first.

To run the demo

```bash
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```





## Training Model

it is a single-gpu version just for personal learning, multi-gpu version please see [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)	



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
  
```bash
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
cd ../..
```

   ```bash
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

   

6. Run script to train and test model

   ```bash
   ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
   # GPU_ID is the GPU you want to test on
   # NET in {vgg16, res50, res101, res152} is the network arch to use
   # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
   # Examples:
   ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
   ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
   ```

   

7. Visualization with Tensorboard

   ```bash
   tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
   tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
   ```

   
   
8. Test and evaluate

   ```bash
   ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
   # GPU_ID is the GPU you want to test on
   # NET in {vgg16, res50, res101, res152} is the network arch to use
   # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
   # Examples:
   ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
   ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
   ```

   You can use `tools/reval.sh` for re-evaluation

   

   
