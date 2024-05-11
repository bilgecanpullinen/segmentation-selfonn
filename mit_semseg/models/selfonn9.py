import torch.nn as nn
import math
from .utils import load_url
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from mit_semseg.lib.utils import as_numpy
from mit_semseg.utils import intersectionAndUnion
from mit_semseg.models.fastonn.osl import *
from mit_semseg.models.fastonn.utils import *
from mit_semseg.models.fastonn import OpTier
from mit_semseg.models.fastonn import SelfONN2d, Trainer
from mit_semseg.models.fastonn.SelfONN import SelfONN2d
BatchNorm2d = SynchronizedBatchNorm2d


class SelfONN9(nn.Module):

    def __init__(self, num_classes=1000):
        self.inplanes = 128
        super(SelfONN9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.Tanh()
        self.conv2 = SelfONN2d(64, 64,kernel_size=3, stride=1, padding=1, bias=False, q=2)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.Tanh()
        self.conv3 = SelfONN2d(64, 64,kernel_size=3, stride=1, padding=1, bias=False, q=2)
        self.bn3 = BatchNorm2d(64)
        self.relu3 = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 =SelfONN2d(64, 128,kernel_size=3,stride=2, padding=1, bias=False, q=2)
        self.bn4 = BatchNorm2d(128)
        self.relu4 = nn.Tanh()
        self.conv5 = SelfONN2d(128, 128,kernel_size=3,stride=1, padding=1, bias=False, q=2)
        self.bn5 = BatchNorm2d(128)
        self.relu5 = nn.Tanh()
        self.conv6 = SelfONN2d(128, 256,kernel_size=3,stride=2, padding=1, bias=False, q=2)
        self.bn6 = BatchNorm2d(512)
        self.relu6 = nn.Tanh()
        self.conv7 = SelfONN2d(256, 256,kernel_size=3,stride=1, padding=1, bias=False, q=2)
        self.bn7 = BatchNorm2d(512)
        self.relu7 = nn.Tanh()
        self.conv8 = SelfONN2d(256, 512,kernel_size=3,stride=2, padding=1, bias=False, q=2)
        self.bn8 = BatchNorm2d(512)
        self.relu8 = nn.Tanh()
        self.conv9 = SelfONN2d(512, 512,kernel_size=3,stride=1, padding=1, bias=False, q=2)
        self.bn9 = BatchNorm2d(512)
        self.relu9 = nn.Tanh()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.relu9(self.bn9(self.conv9(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def selfonn9(pretrained=False, **kwargs):
    model = SelfONN9(**kwargs)
    return model
