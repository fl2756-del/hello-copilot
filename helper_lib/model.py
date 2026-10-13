import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------- FCNN (from Module_3_Practical_1_FCNN notebook) ----------
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------- CNN (Assignment 2 architecture, 64x64x3 input) ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # Input channels = 3, Output channels = 16 -> 16 x 64 x 64
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # Input channels = 16, Output channels = 32 -> 32 x 32 x 32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer with 100 units
        self.fc2 = nn.Linear(100, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # -> 32 x 16 x 16
        x = x.view(-1, 32 * 16 * 16)  # Flatten -> 8192
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- EnhancedCNN (from Module_3_Practical_3_CNN notebook) ----------
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name):
    # define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model