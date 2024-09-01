import torch
import torch.nn as nn

class PointWiseCNN(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_prob=0.1):
        super(PointWiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(p=dropout_prob / 5)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(p=dropout_prob)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.conv4(x)
        return x
