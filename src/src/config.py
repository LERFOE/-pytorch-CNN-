#Here are the contents for the specified files:

#/pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/models.py**

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Assuming input image size is 224x224
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#**/pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/config.py**

class Config:
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 20
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'