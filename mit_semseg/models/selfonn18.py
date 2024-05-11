import torch.nn as nn
import math
from .utils import load_url
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from mit_semseg.models.fastonn.osl import *
from mit_semseg.models.fastonn.utils import *
from mit_semseg.models.fastonn import OpTier
from mit_semseg.models.fastonn import SelfONN2d, Trainer
from mit_semseg.models.fastonn.SelfONN import SelfONN2d
BatchNorm2d = SynchronizedBatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = SelfONN2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, q=2)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.Tanh()
        self.conv2 = SelfONN2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, q=2)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=2, downsample=None):
        super().__init__()
        self.conv1 = SelfONN2d(inplanes, planes, kernel_size=1, stride=2, padding=1, bias=False, q=2)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = SelfONN2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, q=2)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = SelfONN2d(planes, planes * 4, kernel_size=1, stride=2, padding=1,bias=False, q=2)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.Tanh()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SelfONN18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super().__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.Tanh()
        self.conv2 = SelfONN2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, q=2)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.Tanh()
        self.conv3 = SelfONN2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, q=2)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, (1. / n)) #does not have relu applied on its input
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SelfONN2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False, q=2),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def selfonn18(pretrained=False, **kwargs):
    model = SelfONN18(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

