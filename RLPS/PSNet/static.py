import torch 
import torch.nn as nn
from thop import clever_format, profile
import torch.nn.functional as F

class PSNet(nn.Module):
    def __init__(self, input_channel, k=2):
        super(PSNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, [1, 1])
        self.conv2 = nn.Conv2d(64, 64, [1, 1])
        self.conv3 = nn.Conv2d(64, 64, [1, 1])
        self.conv4 = nn.Conv2d(64, 128, [1, 1])
        self.conv5 = nn.Conv2d(128, 1024, [1, 1])
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = torch.sum(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.bn6(self.fc1(x)))
        # x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.bn7(self.fc2(x)))
        # x = F.dropout(x, 0.3, training=self.training)
        x = self.fc3(x)
        return F.softmax(x, dim=1)

band = 200
classes = 16

model = PSNet(band, classes)
inputs = torch.randn(1, band, 625, 1)
flops, params = profile(model, inputs=(inputs, ))
flops, params = clever_format([flops, params], "%.2f")
print(flops)
print(params)