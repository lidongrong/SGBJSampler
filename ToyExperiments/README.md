# General Introduction
This folder replicates the toy experiments, by testing sampler on different toy distributions.

# Scripts Descriptions
**Toy.py**: Replication of results on Gaussian posterior.

**ToyT.py**: Replication of results on T posterior.

**ToyMultiModal.py**: Replication of results on a simple multi-modal posterior.

# Replications

Results summarized from different experiments can be founded below.

## Gaussian Distribution
Performance of the proposed method on a Gaussian density under different step sizes compared to other samplers can be found in this section.

### Step Size = 0.1
![avatar](images/Step1e-1.png)

### Step Size = 0.05
![avatar](images/Step5e-2.png)

### Step Size = 0.01
![avatar](images/Step1e-2.png)

### Step Size = 0.005
![avatar](images/Step5e-3.png)

### Step Size = 0.001
![avatar](images/Step1e-3.png)

### Step Size = 0.0005
![avatar](images/Step5e-4.png)

## Heavy-Tailed T Distribution
Performance of the proposed method on a heavy-tailed T density under different step sizes compared to other samplers can be found in this section.

### Step Size = 0.1
![avatar](images/1e-1.png)

### Step Size = 0.05
![avatar](images/5e-2.png)

### Step Size = 0.01
![avatar](images/1e-2.png)

### Step Size = 0.005
![avatar](images/5e-3.png)

### Step Size = 0.001
![avatar](images/1e-3.png)

### Step Size = 0.0005
![avatar](images/5e-4.png)


## Gaussian Mixture Distribution
Trace plot of the proposed method on a two-mode Gaussian mixture distribution under different step sizes compared to other samplers can be found in this section.

![avatar](images/trace_plots.png)
