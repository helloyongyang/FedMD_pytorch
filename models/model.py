import torch.nn as nn
import torch.nn.functional as F


class Net_2_layer_CNN(nn.Module):
    def __init__(self, n1, n2, dropout_rate, n_classes):
        super(Net_2_layer_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(n1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(n2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(n2, n_classes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.pool1(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))

        return out



class Net_3_layer_CNN(nn.Module):
    def __init__(self, n1, n2, n3, dropout_rate, n_classes):
        super(Net_3_layer_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(n1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(n2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(n3)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(n3, n_classes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.pool1(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        out = self.pool2(out)

        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dropout3(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))

        return out
