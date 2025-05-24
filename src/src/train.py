#Contents of /pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/models.py:

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

#Contents of /pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/train.py:

import torch
import torch.optim as optim
import torch.nn.functional as F
from src import CNNModel
from .utils import get_data_loaders, plot_loss_accuracy
from .config import Config

def train():
    config = Config()
    train_loader, val_loader = get_data_loaders(config.batch_size)
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    plot_loss_accuracy(train_losses, train_accuracies)

if __name__ == '__main__':
    train()