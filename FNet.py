<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNet(nn.Module):
    def __init__(self, input_dim, ncluster):
        super(FNet, self).__init__()

        self.conv1_1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, ncluster, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(ncluster)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3_1(x))
        x = self.bn3(x)

        x = F.relu(self.conv4_1(x))
        x = self.bn4(x)

        x = F.relu(self.conv5_1(x))
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.bn6(x)

        return x
=======
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNet(nn.Module):
    def __init__(self, input_dim, ncluster):
        super(FNet, self).__init__()

        self.conv1_1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, ncluster, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(ncluster)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3_1(x))
        x = self.bn3(x)

        x = F.relu(self.conv4_1(x))
        x = self.bn4(x)

        x = F.relu(self.conv5_1(x))
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.bn6(x)

        return x
>>>>>>> 82d4fd17cb121552baddef5e429baf146d1133c1
