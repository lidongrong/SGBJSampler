# General Introduction
This folder contains scripts that replicate the CIFAR-10 experiments.

# Scripts Description
**CIFAR10net.py** contains neural networks that trains and conducts inference on the dataset.

**CIFAR10test.py** contains the training and evaluation procedure.

**optimizers.py** contains the samplers. Don't modify otherwise may lead to unexpected errors.

**schedulers.py** contains step size schedulers.

# Replication
Experiments on proposed method VS existing methods can be replicated by above scripts. Metrics used to evaluate is the posterior prediction acc among them. Results on mean posterior prediction acc and corresponding std see below.

## 25% of the training process
![avatar](images/CIFAR_64.png)

## 50% of the training process
![avatar](images/CIFAR_128.png)

## 75% of the training process
![avatar](images/CIFAR_192.png)

## 100% of the training process
![avatar](images/CIFAR_249.png)

