# Project - Facial Keypoint Detection

In this project, I have used the dataset of image-caption pairs to train a CNN-RNN model to automatically generate images from captions


### 1) The Dataset

The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

More about the [dataset](http://cocodataset.org/#home).

Example Image:

<img src="images/image1.png"/>


### 2) The Neural Network
The Network is a combination of Enocder and Decoder.

The encoder uses the pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.

<img src="images/enocder.png"/>


The decoder architecture is based on this [paper](https://arxiv.org/pdf/1411.4555.pdf)

<img src="images/decoder.png"/>

The Complete Architechture :



<img src="images/encoder-decoder.png"/>


### 3) Traning the Network

#### Question: How did you select the trainable parameters of your architecture? Why do you think this is a good choice?

Answer: I went with the same : list(decoder.parameters()) + list(encoder.embed.parameters())

Encoder was Resnet50 which was already trained (Transfer Learning).But the last layer of the Encoder needed to be trained, so that it's output makes sense to the following RNN Decoder.

Decoder was completely new ,So had to train all the layers.

### Question: How did you select the optimizer used to train your model?

Answer: I chose ADAM Optimizer with a learning rate of 0.001

1) Since it combines both RMS Prop and Momentum.

2) I had used the same in the Previous CNN Project ( Facial Keypoints ) and It provided good results and the training seemed fast enough.

So decided to go for the same with the same learning rate.

### 4) Results

The trained network was then tested on unseen images and here are the results :

<img src="images/image2.png"/> <img src="images/image4.png"/>
<img src="images/image3.png"/> <img src="images/image5.png"/>
















Feature tracking part and  various detector / descriptor combinations were tested  to see which ones perform best. This project consists of four parts:

* First   ,loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Second  ,integrating several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and comparing them with regard to number of keypoints and speed. 
* Third   ,descriptor extraction and matching using brute force and also the FLANN approach. 
* Finally , Testing the various algorithms in different combinations and compare them with regard to some performance measures. 


## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.
