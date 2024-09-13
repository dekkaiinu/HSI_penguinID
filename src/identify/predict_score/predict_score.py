import torch
import torch.nn as nn

class GetScoreModule(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(GetScoreModule, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor, weight_mask: torch.Tensor):
        output = self.model(x)
        softmax_output = nn.functional.softmax(output, dim=1)
        masked_output = softmax_output * weight_mask.unsqueeze(1).expand_as(softmax_output)
        # global average pooling
        output = torch.sum(masked_output, dim=(2, 3))
        return output
    


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


if __name__ == "__main__":
    test = GetScoreModule(model=PointWiseCNN(input_channels=151, output_channels=16, dropout_prob=0.5))

    test_data = torch.randn(1, 151, 64, 64)
    test_weight = torch.randn(1, 64, 64)

    print(test_data.shape)
    print(test_weight.shape)

    test_output = test(test_data, test_weight)
    print(test_output.shape)
