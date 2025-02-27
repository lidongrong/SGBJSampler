import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models


class VGGStyleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGStyleCNN, self).__init__()

        # Convolutional block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 2
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 3
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Convolutional block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)

        # Convolutional block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)

        # Convolutional block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        # Initialize weights using He uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer input 32x32x3, output 28x28x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        # Second convolutional layer input 14x14x32, output 10x10x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Third convolutional layer input 5x5x64, output 3x3x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # Fully connected layer 3x3x128 inputs, 10 outputs (for 10 classes)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten image input
        x = x.view(-1, 128 * 3 * 3)
        # Fully connected layer
        x = self.fc(x)
        # Softmax on the outputs
        x = F.log_softmax(x, dim=1)
        return x

class ResNetForCIFAR10(models.ResNet):
    def __init__(self, num_classes=10):
        super(ResNetForCIFAR10, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.maxpool = nn.Identity()  # Remove max-pooling layer


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR10, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)

        # Modify the first convolutional layer and the fully connected layer
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove the maxpool layer
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

def ResNet50CIFAR10(num_classes=10):
    # Load a pre-defined ResNet50 model with a custom number of output classes
    # Replace the first convolutional layer to be suitable for 32x32 images
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # CIFAR-10 images are too small for the initial max pool
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes in CIFAR-10
    return model

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

# Upsample layer
class UpsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsampleBlock, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.upsample(x)

# Downsample layer
class DownsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownsampleBlock, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.downsample(x)

# Generator
class CIFARGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape=(3,32,32)):
        super(CIFARGenerator, self).__init__()
        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            UpsampleBlock(128, 128),
            ResidualBlock(128),
            UpsampleBlock(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator
class CIFARDiscriminator(nn.Module):
    def __init__(self, img_shape=(3,32,32)):
        super(CIFARDiscriminator, self).__init__()

        self.model = nn.Sequential(
            DownsampleBlock(img_shape[0], 64),
            ResidualBlock(64),
            DownsampleBlock(64, 128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity