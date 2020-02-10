# ERNet - Segmentation of Endoplasmic Reticulum microscopy images
Meng Lu<sup>1</sup>, Francesca W. van Tartwijk<sup>1</sup>, Julie Qiaojin Lin<sup>1</sup>, Wilco Nijenhuis, Pierre Parutto, Marcus Fantham<sup>1</sup>,  __Charles N. Christensen<sup>1,*</sup>__, Edward  Avezov, Christine E. Holt, Alan Tunnacliffe, David Holcman, Lukas C. Kapitein, Gabriele Kaminski Schierle<sup>1</sup>, Clemens F. Kaminski<sup>1</sup></br>
<sup>1</sup>University of Cambridge, Department of Chemical Engineering and Biotechnology</br>
<sup> *</sup>Author of this repository - GitHub username: [charlesnchr](http://github.com/charlesnchr) - Email address: <code>charles.n.chr@gmail.com</code>

Paper with results based on the code in this repository: https://www.biorxiv.org/content/10.1101/2020.01.15.907444v2

## Introduction 

ERNet is an artificial neural network based model for segmentation of endoplasmic reticulum microscopy images. The network architecture of choice is a deep residual network inspired by EDSR and RCAN ([Lim et al. 2017](https://arxiv.org/abs/1707.02921), [Zhang et al., 2018](https://arxiv.org/abs/1807.02758)). The modified architecture is shown in the image below with a block corresponding to EDSR. 

<img src="fig/architecture.png">

The architectures of these models are among several residual learning networks (cite ResNet, SRGan, EDSR) designed for image restoration, specifically single image super-resolution (SR), i.e. image upsampling. The state-of-the-art SR architectures generally do not use downsampling between layers, but instead make training of deep networks feasible by following the structure of residual networks as first introduced with ResNets intended for image classification. The design idea of residual networks was taken one step further in EDSR with the proposal of a modified residual building block called ResBlock, which was found to be superior to the previously proposed and more directly adapted ResNet model called SRResNet

Choosing the first part of our segmentation model to have an architecture built for restoration ensures that it is capable of handling low signal-to-noise ratio as it can learn to perform denoising in these early layers of its network. A neural network model intended for image restoration will by default perform regression in order to output pixel value predictions in the same colour space as the input image. This is achieved during model training by minimising an appropriate loss function, typically the mean squared error. 

## Installation

This implementation requires Pytorch. We have tested ERNet with Python 3.6 and 3.7 and Pytorch 1.2 and 1.4. 

## Training 
:coffee:
* Start in the root folder: ```./```
* Run the command:
  ```
  python run.py --root trainingdata/partitioned_256 --imageSize 256 --out 0206_ERNet_rcan-rg5 --model rcan --nch_in 1 --nch_out 2 --ntrain 480 --ntest 20 --scale 1 --task segment --batchSize 2 --n_resgroups 5 --n_resblocks 10 --lr 0.0001 --scheduler 20,0.5 --nepoch 100 --dataset pickledataset
  ```

