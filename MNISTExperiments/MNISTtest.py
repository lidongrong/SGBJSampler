import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers import SGLDOptimizer,SGBOptimizer, SGHMCOptimizer, SGNHTOptimizer
from MNISTnet import MLP
from torch.optim import lr_scheduler
from regularizer import l2_regularization
import pandas as pd
import numpy as np
from schedulers import csg_mcmc_scheduler, polynomial

"""
This script compares the performance of sampling neural networks with different samplers.
"""

# Check if CUDA is available and set the device accordingly
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# MNIST Dataset transformation
# don't adjust this, chatGPT tells me to work like this, I don't know why it chooses these parameters
mnist_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training and test data for MNIST
# select batch size as 128, can also use other batch sizes
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=mnist_transforms)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=mnist_transforms)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)


# Define the loss function & prior parameter
criterion = nn.CrossEntropyLoss()
lambda_reg = 1


# Training the network
def train(model, device, train_loader, optimizer, scheduler,epoch):
    model.train()
    N = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = N * criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % 100 == 0:
            print(f'using optimizer {optimizer.__class__.__name__} with lr {optimizer.param_groups[0]["lr"]}')
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]')

# Evaluate the network on the test set
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() + l2_regularization(model = model, lambda_reg = lambda_reg,device = device) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return correct/len(test_loader.dataset)

# the enire training & evaluate (OOS evaluation) procedure
def evaluate(model,device,train_loader,test_loader,optimizer,scheduler,epochs):
    stat=[]
    # Train and test the model for epochs
    for epoch in range(1, epochs):
        train(model, device, trainloader, optimizer,scheduler, epoch)
        acc = test(model, device, testloader)
        stat.append(acc)
    return stat

# The epochs we record the acc, record them at 25%, 50%, 75% and 100%
check_points = [2,5,7,9]
steps = [1e-1,0.075,0.05,0.025,0.01,0.0075,0.005,0.0025,0.001,
         0.00075,0.0005,0.00025,0.0001,0.000075,0.00005,0.000025,0.00001,
        0.0000075,0.000005,0.0000025,0.000001]
results = []



for step in steps:
    # check point for the proposed sampler
    sgb_ckpt = {ckpt:[] for ckpt in check_points}
    # check points for sgld
    sgld_ckpt = {ckpt:[] for ckpt in check_points}
    # check points for sgnht
    sgnht_ckpt = {ckpt:[] for ckpt in check_points}
    # check points for sghmc
    sghmc_ckpt = {ckpt:[] for ckpt in check_points}
    
    for k in range(1):
        # repeat 20 experiments (for each sampler at each step size level), more can be done if you like
        print(f'experiment {k} with step size {step}')
        bmodel = MLP().to(device)
        lmodel = MLP().to(device)
        hmodel = MLP().to(device)
        nmodel = MLP().to(device)

        sgb_optimizer = SGBOptimizer(bmodel.parameters(), lr=step)
        sgld_optimizer = SGLDOptimizer(lmodel.parameters(), lr=step)
        sghmc_optimizer = SGHMCOptimizer(hmodel.parameters(),lr = step)
        sgnht_optimizer = SGNHTOptimizer(nmodel.parameters(),lr = step)

        # learning rate scheduler
        # I used decaying learning rate. However, other learning rates (e.g. cosine) produce similar results
        lr_lambda = lambda k:  1 / (( 1 * (k+1))**0.5)
        # Create the scheduler
        sgb_scheduler = lr_scheduler.LambdaLR(sgb_optimizer, lr_lambda)
        sgld_scheduler = lr_scheduler.LambdaLR(sgld_optimizer, lr_lambda)
        sghmc_scheduler = lr_scheduler.LambdaLR(sghmc_optimizer, lr_lambda)
        sgnht_scheduler = lr_scheduler.LambdaLR(sgnht_optimizer, lr_lambda)

        # inference
        sghmc_stat = evaluate(hmodel, device, trainloader, testloader, optimizer=sghmc_optimizer,
                               scheduler=sghmc_scheduler, epochs=11)
        sgnht_stat = evaluate(nmodel, device, trainloader, testloader, optimizer=sgnht_optimizer,
                                 scheduler=sgnht_scheduler, epochs=11)
        sgb_stat=evaluate(bmodel,device,trainloader,testloader,
                             optimizer = sgb_optimizer,scheduler = sgb_scheduler,epochs = 11)
        sgld_stat = evaluate(lmodel,device,trainloader,testloader,
                                 optimizer = sgld_optimizer,scheduler = sgld_scheduler,epochs = 11)

        for ckpt in check_points:
            sgb_ckpt[ckpt].append(np.mean(sgb_stat[max(0,ckpt-2):ckpt]))
            sgld_ckpt[ckpt].append(np.mean(sgld_stat[max(0,ckpt-2):ckpt]))
            sghmc_ckpt[ckpt].append(np.mean(sghmc_stat[max(0,ckpt-2):ckpt]))
            sgnht_ckpt[ckpt].append(np.mean(sgnht_stat[max(0,ckpt-2):ckpt]))

    for ckpt in check_points:
        results.append({
            'Sampler': f'SGB step size={step}',
            'Checkpoint':ckpt,
            'ACC': np.mean(sgb_ckpt[ckpt]),
            'std': np.std(sgb_ckpt[ckpt])
        })
        results.append({
            'Sampler': f'SGLD step size={step}',
            'Checkpoint':ckpt,
            'ACC': np.mean(sgld_ckpt[ckpt]),
            'std': np.std(sgld_ckpt[ckpt])
        })
        results.append({
            'Sampler': f'SGHMC step size={step}',
            'Checkpoint': ckpt,
            'ACC': np.mean(sghmc_ckpt[ckpt]),
            'std': np.std(sghmc_ckpt[ckpt])
        })
        results.append({
            'Sampler': f'sgnht step size={step}',
            'Checkpoint': ckpt,
            'ACC': np.mean(sgnht_ckpt[ckpt]),
            'std': np.std(sgnht_ckpt[ckpt])
        })

results=pd.DataFrame(results)
#results.to_csv('MNIST_cyclical_results.txt')
# specify the saving path you like
results.to_csv('MNIST_decay_results.txt')
