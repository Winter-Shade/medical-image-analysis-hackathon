import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=13):
        super(CustomResNet18, self).__init__()
        resnet18 = models.resnet18(weights=None)

        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18.maxpool = nn.Identity()

        resnet18.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        resnet18.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        resnet18.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        resnet18.fc = nn.Linear(512, num_classes)

        self.model = resnet18

    def forward(self, x):
        return self.model(x)
