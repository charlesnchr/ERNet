<h1 align="center">ERNet</h1>

<p align="center">
  <b>A super-resolution network, rebuilt to trace the endoplasmic reticulum.</b><br>
  It never downsamples, so tubules a few pixels wide come out in one piece.<br>
  This is the original CNN. The transformer that followed it is <a href="https://github.com/charlesnchr/ERnet-v2">ERnet-v2</a>.
</p>

<p align="center">
  <a href="https://doi.org/10.1126/sciadv.abc7209"><img src="https://img.shields.io/badge/Science%20Advances-2020-b31b1b" alt="Paper in Science Advances"></a>
  <a href="https://doi.org/10.1101/2020.01.15.907444"><img src="https://img.shields.io/badge/bioRxiv-2020.01.15.907444-b31b1b" alt="bioRxiv preprint"></a>
  <a href="https://github.com/charlesnchr/ERnet-v2"><img src="https://img.shields.io/badge/successor-ERnet--v2-8250df" alt="Successor: ERnet-v2"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="GPL-3.0 license"></a>
</p>

<p align="center">
  <img src="fig/hero.gif" alt="A wide-field image of a cell and a SIM image of the endoplasmic reticulum wiping to their segmentation and then to the skeleton graph of the tubules" width="900">
</p>
<p align="center">
  <sub>A live COS-7 cell under a wide-field microscope and a SIM image of the ER, segmented and converted to a graph by the two models in this repository. Run <code>python fig/make_figures.py</code> to reproduce it.</sub>
</p>

## Introduction

ERNet is an artificial neural network based model for segmentation of endoplasmic reticulum microscopy images. The network architecture of choice is a deep residual network inspired by EDSR and RCAN ([Lim et al. 2017](https://arxiv.org/abs/1707.02921), [Zhang et al., 2018](https://arxiv.org/abs/1807.02758)). The modified architecture is shown in the image below with a block corresponding to EDSR.

<img src="fig/architecture.png">

The architectures of these models are among several residual learning networks (cite ResNet, SRGan, EDSR) designed for image restoration, specifically single image super-resolution (SR), i.e. image upsampling. The state-of-the-art SR architectures generally do not use downsampling between layers, but instead make training of deep networks feasible by following the structure of residual networks as first introduced with ResNets intended for image classification. The design idea of residual networks was taken one step further in EDSR with the proposal of a modified residual building block called ResBlock, which was found to be superior to the previously proposed and more directly adapted ResNet model called SRResNet

Choosing the first part of our segmentation model to have an architecture built for restoration ensures that it is capable of handling low signal-to-noise ratio as it can learn to perform denoising in these early layers of its network. A neural network model intended for image restoration will by default perform regression in order to output pixel value predictions in the same colour space as the input image. This is achieved during model training by minimising an appropriate loss function, typically the mean squared error.

## Installation :wrench:

This implementation requires Pytorch. We have tested ERNet with Python 3.6 and 3.7 and Pytorch 1.2 and 1.4.

## Training data :bar_chart:
A simple training data set is provided in `./trainingdata/partitioned_256`. This dataset is preprocessed based on the labelled datafiles in `./trainingdata/labelled_data` using the command:
```powershell
python buildTrainingData.py
```

## Model training :watch:


* Start in the root folder: ```./```
* See available command line options with `python run.py -h`
* An example of options for training with RCAN could be:
  ```powershell
  --root trainingdata/partitioned_256
  --out trained_models/ERNet_rcan-rg5
  --imageSize 256
  --model rcan
  --nch_in 1
  --nch_out 2
  --ntrain 480
  --ntest 20
  --batchSize 2
  --n_resgroups 5
  --n_resblocks 10
  --lr 0.0001
  --scheduler 20,0.5
  --nepoch 100
  --dataset pickledataset
  ```
* Run the `run.py` script with Python, e.g.:
  ```powershell
  python run.py --root trainingdata/partitioned_256 --out trained_models/ERNet_rcan-rg5 --imageSize 256 --model rcan --nch_in 1 --nch_out 2 --ntrain 480 --ntest 20 --batchSize 2 --n_resgroups 5 --n_resblocks 10 --lr 0.0001 --scheduler 20,0.5 --nepoch 100 --dataset pickledataset
  ```

## Testing
A couple of test images of the endoplasmic reticulum are provided in `testdata/input`, for instance:

<img src="fig/testimg.png" width=200>

To evaluate a model on these images in order to get the segmentation maps, use the script `evaluate.py`.

 A simple pretrained model based on the example in the 'Model training' section is in the folder `pretrained_model`.

__Evaluating inputs with a model:__
* Start in the root folder: ```./```
* See available command line options with `python evaluate.py -h`
* An example of applying the pretrained model on the provided test images could be:
  ```powershell
  --root testdata/input
  --weights pretrained_model
  --imageSize 1000
  --out testdata/output
  --model rcan
  --nch_in 1
  --nch_out 2
  --n_resgroups 5
  --n_resblocks 10
  ```
* Run the `evaluate.py` script with Python, e.g.:
  ```powershell
  python evaluate.py --root testdata/input --weights pretrained_model --imageSize 1000 --out testdata/output --model rcan --nch_in 1 --nch_out 2 --n_resgroups 5 --n_resblocks 10
  ```

The above example generates the output in `testdata/output`, such as:

 <img src="fig/testoutput.png" width=200>
