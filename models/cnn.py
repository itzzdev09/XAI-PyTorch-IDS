import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()

        # Expect input shape: (batch_size, input_dim)
        self.input_dim = input_dim

        # 1D convolution over features
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Reshape input for CNN: (batch_size, channels=1, features)
        x = x.view(-1, 1, self.input_dim)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
