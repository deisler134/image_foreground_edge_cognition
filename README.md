# Deep Image Matting
This repository is to reproduce Deep Image Matting.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow 1.9.0](https://www.tensorflow.org/)
- [Keras 2.1.6](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset
### Adobe Deep Image Matting Dataset
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact author for the dataset.

### MSCOCO
Go to [MSCOCO](http://cocodataset.org/#download) to download:
* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip)


### PASCAL VOC
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
* VOC challenge 2008 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)

## ImageNet Pretrained Models
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into models folder.


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
Download pre-trained Deep Image Matting [Model](https://github.com/foamliu/Deep-Image-Matting/releases/download/v1.0/final.42-0.0398.hdf5) to "models" folder then run:
```bash
$ python demo.py
```

