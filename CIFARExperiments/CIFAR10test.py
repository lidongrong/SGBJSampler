import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers import SGLDOptimizer,SGBJOptimizer
from CIFAR10net import ConvNet, ResNet18CIFAR10, ResNet50CIFAR10, VGGStyleCNN # Adjust the import according to your CIFAR-10 model
from torch.optim import lr_scheduler
from regularizer import l2_regularization
import numpy as np

# set seed
torch.manual_seed(3407)

# Check if CUDA is available and set the device accordingly
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# CIFAR-10 Dataset transformation
cifar10_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust normalization for CIFAR-10
])

# Download and load the training and test data for CIFAR-10
trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=cifar10_transforms)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=cifar10_transforms)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# Initialize the network (ensure that the network is suitable for CIFAR-10, i.e., accepts 3-channel images)
model = ResNet18CIFAR10()  # Use a model appropriate for CIFAR-10

# Define the loss function
criterion = nn.CrossEntropyLoss()
lambda_reg = 1

# Initialize the SGLD optimizer
barker_optimizer = SGBJOptimizer(model.parameters(), lr=1e-6)
langevin_optimizer = SGLDOptimizer(model.parameters(),lr=1e-6)

# learning rate scheduler
lr_lambda = lambda k: 1#1 / (( 1 * (k+1))**0.5)
# Create the scheduler
barker_scheduler = lr_scheduler.LambdaLR(barker_optimizer, lr_lambda)
langevin_scheduler = lr_scheduler.LambdaLR(langevin_optimizer,lr_lambda)

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
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluate the network on the test set
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() #+ l2_regularization(model = model, lambda_reg = lambda_reg) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return correct/len(test_loader.dataset)

def evaluate(model,device,train_loader,test_loader,optimizer,scheduler,epochs):
    stat=[]
    # Train and test the model for epochs
    for epoch in range(1, epochs):
        train(model, device, trainloader, optimizer,scheduler, epoch)
        acc = test(model, device, testloader)
        stat.append(acc)
    return stat

check_points = [20,40,60,79]
steps = [0.1,0.001,0.0001,0.00005,0.00001,
        0.0000075,0.000005,0.0000025,0.000001]
results = []

#barker_stat=evaluate(model,device,trainloader,testloader,optimizer = barker_optimizer,scheduler = barker_scheduler,epochs = 11)


for step in steps:
    barker_ckpt = {ckpt:[] for ckpt in check_points}
    langevin_ckpt = {ckpt:[] for ckpt in check_points}
    for k in range(4):
        bmodel = ResNet18CIFAR10().to(device)
        lmodel = ResNet18CIFAR10().to(device)
        barker_optimizer = SGBJOptimizer(bmodel.parameters(), lr=step)
        langevin_optimizer = SGLDOptimizer(lmodel.parameters(), lr=step)

        # learning rate scheduler
        lr_lambda = lambda k:  1 / (( 1 * (k+1))**0.5)
        # Create the scheduler
        langevin_scheduler = lr_scheduler.LambdaLR(langevin_optimizer, lr_lambda)
        barker_scheduler = lr_scheduler.LambdaLR(barker_optimizer, lr_lambda)


        # inference
        barker_stat = evaluate(bmodel, device, trainloader, testloader, optimizer=barker_optimizer,
                               scheduler=barker_scheduler, epochs=81)
        langevin_stat = evaluate(lmodel, device, trainloader, testloader, optimizer=langevin_optimizer,
                                 scheduler=langevin_scheduler, epochs=81)



        for ckpt in check_points:
            barker_ckpt[ckpt].append(barker_stat[ckpt])
            langevin_ckpt[ckpt].append(langevin_stat[ckpt])

    for ckpt in check_points:
        results.append({
            'Sampler': f'Barker step size={step}',
            'Checkpoint':ckpt,
            'ACC': np.mean(barker_ckpt[ckpt]),
            'std': np.std(barker_ckpt[ckpt])
        })
        results.append({
            'Sampler': f'Langevin step size={step}',
            'Checkpoint':ckpt,
            'ACC': np.mean(langevin_ckpt[ckpt]),
            'std': np.std(langevin_ckpt[ckpt])
        })