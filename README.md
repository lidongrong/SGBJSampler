# SGBJSampler

## General Description

This repository provides the implementation of some of the sampler (optimizers) proposed in my academic research. Sampler is implemented in pure pytorch, which can directly be plugged-in to posterior sampling on complicated models such as Bayesian neural network (BNN).

No redundant parameter tuning required, just plug it in like you are using SGD or the Adam optimizer.

## Experiments
The experiments of the proposed sampler have been implemented on various tasks such as simple Gaussian / T / multi-modal distributions, image classification on MNIST and CIFAR-10 datasets. Results suggest it has a more stable performance under step size tuning when compared to other popular samplers.

Code & Results on various toy models can be found in the folder **ToyExperiments**. Code & Results on image classification on the MNIST dataset can be found in the folder **MNISTExperiments**. Code & Results on image classification on the CIFAR-10 dataset can be found in the folder **CIFARExperiments**.


