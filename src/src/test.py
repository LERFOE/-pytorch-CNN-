Here are the contents for the specified files:

**/pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/models.py**

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Assuming input image size is 224x224
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

**/pytorch-chinese-medicine-classifier/pytorch-chinese-medicine-classifier/src/test.py**

import torch
from models import CNNModel
from utils import load_data, calculate_accuracy

def test_model(model_path, test_loader):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    test_loader = load_data('path_to_test_data')  # Replace with actual test data path
    test_model('path_to_trained_model.pth', test_loader)  # Replace with actual model path