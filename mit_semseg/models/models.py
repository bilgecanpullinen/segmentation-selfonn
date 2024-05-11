import torch
import torch.nn as nn
from . import resnet, selfonn5, selfonn9, selfonn16, selfonn18, cnn5, cnn9, cnn16
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from mit_semseg.lib.utils import as_numpy
from mit_semseg.utils import intersectionAndUnion
from mit_semseg.models.fastonn.osl import *
from mit_semseg.models.fastonn.utils import *
from mit_semseg.models.fastonn import OpTier
from mit_semseg.models.fastonn import SelfONN2d, Trainer
from mit_semseg.models.fastonn.SelfONN import SelfONN2d
BatchNorm2d = SynchronizedBatchNorm2d

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super().__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        intersection, union, area_pred, area_lab = intersectionAndUnion(preds.cpu(), label.cpu(), 150)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        intersection = torch.from_numpy(intersection).cuda().float()
        union = torch.from_numpy(union).cuda().float()
        area_pred = torch.from_numpy(area_pred).cuda().float()
        area_lab = torch.from_numpy(area_lab).cuda().float()
        return acc, intersection, union, area_pred, area_lab

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super().__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        if type(feed_dict) is list:
            feed_dict = feed_dict[0]
            if torch.cuda.is_available():
                feed_dict['img_data'] = feed_dict['img_data'].cuda()
                feed_dict['seg_label'] = feed_dict['seg_label'].cuda()
            else:
                raise RunTimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')

        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])

            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale
            acc, intersection, union, area_pred, area_lab = self.pixel_acc(pred, feed_dict['seg_label'])

            return loss, acc, intersection, union, area_pred, area_lab

        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred

class ModelBuilder:
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_normal_(m.weight, gain=gain)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)
        #selfonn layers are initialized at the fastonn folder

    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = False if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'pspnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=True)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'pspnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=True)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'SelfONNet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=True)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'SelfONNet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=True)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'selfonn5':
            orig_selfonn5 = selfonn5.__dict__['selfonn5'](pretrained=False)
            net_encoder = SelfONN5(orig_selfonn5)
        elif arch == 'cnn5':
            orig_cnn5 = cnn5.__dict__['cnn5'](pretrained=False)
            net_encoder = CNN5(orig_cnn5)
        elif arch == 'selfonn9':
            orig_selfonn9 = selfonn9.__dict__['selfonn9'](pretrained=False)
            net_encoder = SelfONN9(orig_selfonn9)
        elif arch == 'cnn9':
            orig_cnn9 = cnn9.__dict__['cnn9'](pretrained=False)
            net_encoder = CNN9(orig_cnn9)
        elif arch == 'selfonn16':
            orig_selfonn16 = selfonn16.__dict__['selfonn16'](pretrained=False)
            net_encoder = SelfONN16(orig_selfonn16)
        elif arch == 'cnn16':
            orig_cnn16 = cnn16.__dict__['cnn16'](pretrained=False)
            net_encoder = CNN16(orig_cnn16)
        elif arch == 'selfonn18':
            orig_selfonn18 = selfonn18.__dict__['selfonn18'](pretrained=False)
            net_encoder = SelfONN18(orig_selfonn18, dilate_scale=8)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=False)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)

        else:
            raise Exception('Architecture undefined!')

       
        #net_encoder.apply(ModelBuilder.weights_init) #activate this line for using pretrained weights
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c2':
            net_decoder = C2(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c3':
            net_decoder = C3(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c5':
            net_decoder = C5(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup_q3':
            net_decoder = PPMDeepsupQ3(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)

        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.Tanh(),
            )
def conv3x3_bn_self2(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            SelfONN2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False,q=2),
            BatchNorm2d(out_planes),
            nn.Tanh(),
            )
def conv3x3_bn_self3(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            SelfONN2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False,q=3),
            BatchNorm2d(out_planes),
            nn.Tanh(),
            )
def conv3x3_bn_self5(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            SelfONN2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False,q=5),
            BatchNorm2d(out_planes),
            nn.Tanh(),
            )

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()
        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class SelfONN5(nn.Module):
    def __init__(self, orig_selfonn5):
        super(SelfONN5, self).__init__()
        self.conv1 = orig_selfonn5.conv1
        self.bn1 = orig_selfonn5.bn1
        self.relu1 = orig_selfonn5.relu1
        self.conv2 = orig_selfonn5.conv2
        self.bn2 = orig_selfonn5.bn2
        self.relu2 = orig_selfonn5.relu2
        self.conv3 = orig_selfonn5.conv3
        self.bn3 = orig_selfonn5.bn3
        self.relu3 = orig_selfonn5.relu3
        self.maxpool = orig_selfonn5.maxpool
        self.conv4 = orig_selfonn5.conv4
        self.bn4 = orig_selfonn5.bn4
        self.relu4 = orig_selfonn5.relu4
        self.conv5 = orig_selfonn5.conv5
        self.bn5 = orig_selfonn5.bn5
        self.relu5 = orig_selfonn5.relu5


    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

class CNN5(nn.Module):
    def __init__(self, orig_cnn5):
        super(CNN5, self).__init__()

        self.conv1 = orig_cnn5.conv1
        self.bn1 = orig_cnn5.bn1
        self.relu1 = orig_cnn5.relu1
        self.conv2 = orig_cnn5.conv2
        self.bn2 = orig_cnn5.bn2
        self.relu2 = orig_cnn5.relu2
        self.conv3 = orig_cnn5.conv3
        self.bn3 = orig_cnn5.bn3
        self.relu3 = orig_cnn5.relu3
        self.maxpool = orig_cnn5.maxpool
        self.conv4 = orig_cnn5.conv4
        self.bn4 = orig_cnn5.bn4
        self.relu4 = orig_cnn5.relu4
        self.conv5 = orig_cnn5.conv5
        self.bn5 = orig_cnn5.bn5
        self.relu5 = orig_cnn5.relu5


    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

class SelfONN9(nn.Module):
    def __init__(self, orig_selfonn9):
        super(SelfONN9, self).__init__()

        self.conv1 = orig_selfonn9.conv1
        self.bn1 = orig_selfonn9.bn1
        self.relu1 = orig_selfonn9.relu1
        self.conv2 = orig_selfonn9.conv2
        self.bn2 = orig_selfonn9.bn2
        self.relu2 = orig_selfonn9.relu2
        self.conv3 = orig_selfonn9.conv3
        self.bn3 = orig_selfonn9.bn3
        self.relu3 = orig_selfonn9.relu3
        self.maxpool = orig_selfonn9.maxpool
        self.conv4 = orig_selfonn9.conv4
        self.bn4 = orig_selfonn9.bn4
        self.relu4 = orig_selfonn9.relu4
        self.conv5 = orig_selfonn9.conv5
        self.bn5 = orig_selfonn9.bn5
        self.relu5 = orig_selfonn9.relu5
        self.conv6 = orig_selfonn9.conv6
        self.bn6 = orig_selfonn9.bn6
        self.relu6 = orig_selfonn9.relu6
        self.conv7 = orig_selfonn9.conv7
        self.bn7 = orig_selfonn9.bn7
        self.relu7 = orig_selfonn9.relu7
        self.conv8 = orig_selfonn9.conv8
        self.bn8 = orig_selfonn9.bn8
        self.relu8 = orig_selfonn9.relu8
        self.conv9 = orig_selfonn9.conv9
        self.bn9 = orig_selfonn9.bn9
        self.relu9 = orig_selfonn9.relu9


    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        conv_out.append(x)
        x = self.relu6(self.bn6(self.conv6(x)))
        conv_out.append(x)
        x = self.relu7(self.bn7(self.conv7(x)))
        conv_out.append(x)
        x = self.relu8(self.bn8(self.conv8(x)))
        conv_out.append(x)
        x = self.relu9(self.bn9(self.conv9(x)))
        conv_out.append(x)


        if return_feature_maps:
            return conv_out
        return [x]

class CNN9(nn.Module):
    def __init__(self, orig_cnn9):
        super(CNN9, self).__init__()

        self.conv1 = orig_cnn9.conv1
        self.bn1 = orig_cnn9.bn1
        self.relu1 = orig_cnn9.relu1
        self.conv2 = orig_cnn9.conv2
        self.bn2 = orig_cnn9.bn2
        self.relu2 = orig_cnn9.relu2
        self.conv3 = orig_cnn9.conv3
        self.bn3 = orig_cnn9.bn3
        self.relu3 = orig_cnn9.relu3
        self.maxpool = orig_cnn9.maxpool
        self.conv4 = orig_cnn9.conv4
        self.bn4 = orig_cnn9.bn4
        self.relu4 = orig_cnn9.relu4
        self.conv5 = orig_cnn9.conv5
        self.bn5 = orig_cnn9.bn5
        self.relu5 = orig_cnn9.relu5
        self.conv6 = orig_cnn9.conv6
        self.bn6 = orig_cnn9.bn6
        self.relu6 = orig_cnn9.relu6
        self.conv7 = orig_cnn9.conv7
        self.bn7 = orig_cnn9.bn7
        self.relu7 = orig_cnn9.relu7
        self.conv8 = orig_cnn9.conv8
        self.bn8 = orig_cnn9.bn8
        self.relu8 = orig_cnn9.relu8
        self.conv9 = orig_cnn9.conv9
        self.bn9 = orig_cnn9.bn9
        self.relu9 = orig_cnn9.relu9


    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        conv_out.append(x)
        x = self.relu6(self.bn6(self.conv6(x)))
        conv_out.append(x)
        x = self.relu7(self.bn7(self.conv7(x)))
        conv_out.append(x)
        x = self.relu8(self.bn8(self.conv8(x)))
        conv_out.append(x)
        x = self.relu9(self.bn9(self.conv9(x)))
        conv_out.append(x)


        if return_feature_maps:
            return conv_out
        return [x]

class SelfONN16(nn.Module):
    def __init__(self, orig_selfonn16):
        super(SelfONN16, self).__init__()

        self.conv1 = orig_selfonn16.conv1
        self.bn1 = orig_selfonn16.bn1
        self.relu1 = orig_selfonn16.relu1
        self.maxpool = orig_selfonn16.maxpool
        self.conv2 = orig_selfonn16.conv2
        self.bn2 = orig_selfonn16.bn2
        self.relu2 = orig_selfonn16.relu2
        self.conv3 = orig_selfonn16.conv3
        self.bn3 = orig_selfonn16.bn3
        self.conv4 = orig_selfonn16.conv4
        self.bn4 = orig_selfonn16.bn4
        self.relu4 = orig_selfonn16.relu4
        self.conv5 = orig_selfonn16.conv5
        self.bn5 = orig_selfonn16.bn5
        self.conv6 = orig_selfonn16.conv6
        self.bn6 = orig_selfonn16.bn6
        self.relu6 = orig_selfonn16.relu6
        self.conv7 = orig_selfonn16.conv7
        self.bn7 = orig_selfonn16.bn7
        self.conv8 = orig_selfonn16.conv8
        self.bn8 = orig_selfonn16.bn8
        self.conv9 = orig_selfonn16.conv9
        self.bn9 = orig_selfonn16.bn9
        self.relu9 = orig_selfonn16.relu9
        self.conv10 = orig_selfonn16.conv10
        self.bn10 = orig_selfonn16.bn10
        self.conv11 = orig_selfonn16.conv11
        self.bn11 = orig_selfonn16.bn11
        self.relu11 = orig_selfonn16.relu11
        self.conv12 = orig_selfonn16.conv12
        self.bn12 = orig_selfonn16.bn12
        self.conv13 = orig_selfonn16.conv13
        self.bn13 = orig_selfonn16.bn13
        self.conv14 = orig_selfonn16.conv14
        self.bn14 = orig_selfonn16.bn14
        self.relu14 = orig_selfonn16.relu14
        self.conv15 = orig_selfonn16.conv15
        self.bn15 = orig_selfonn16.bn15
        self.conv16 = orig_selfonn16.conv16
        self.bn16 = orig_selfonn16.bn16
        self.relu16 = orig_selfonn16.relu16


    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.bn3(self.conv3(x))
        conv_out.append(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.bn5(self.conv5(x))
        conv_out.append(x)
        x = self.relu6(self.bn6(self.conv6(x)))
        conv_out.append(x)
        x = self.bn7(self.conv7(x))
        conv_out.append(x)
        x = self.bn8(self.conv8(x))
        conv_out.append(x)
        x = self.relu9(self.bn9(self.conv9(x)))
        conv_out.append(x)
        x = self.bn10(self.conv10(x))
        conv_out.append(x)
        x = self.relu11(self.bn11(self.conv11(x)))
        conv_out.append(x)
        x = self.bn12(self.conv12(x))
        conv_out.append(x)
        x = self.bn13(self.conv13(x))
        conv_out.append(x)
        x = self.relu14(self.bn14(self.conv14(x)))
        conv_out.append(x)
        x = self.bn15(self.conv15(x))
        conv_out.append(x)
        x = self.relu16(self.bn16(self.conv16(x)))
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

class CNN16(nn.Module):
    def __init__(self, orig_cnn16):
        super(CNN16, self).__init__()

        self.conv1 = orig_cnn16.conv1
        self.bn1 = orig_cnn16.bn1
        self.relu1 = orig_cnn16.relu1
        self.maxpool = orig_cnn16.maxpool
        self.conv2 = orig_cnn16.conv2
        self.bn2 = orig_cnn16.bn2
        self.relu2 = orig_cnn16.relu2
        self.conv3 = orig_cnn16.conv3
        self.bn3 = orig_cnn16.bn3
        self.conv4 = orig_cnn16.conv4
        self.bn4 = orig_cnn16.bn4
        self.relu4 = orig_cnn16.relu4
        self.conv5 = orig_cnn16.conv5
        self.bn5 = orig_cnn16.bn5
        self.conv6 = orig_cnn16.conv6
        self.bn6 = orig_cnn16.bn6
        self.relu6 = orig_cnn16.relu6
        self.conv7 = orig_cnn16.conv7
        self.bn7 = orig_cnn16.bn7
        self.conv8 = orig_cnn16.conv8
        self.bn8 = orig_cnn16.bn8
        self.conv9 = orig_cnn16.conv9
        self.bn9 = orig_cnn16.bn9
        self.relu9 = orig_cnn16.relu9
        self.conv10 = orig_cnn16.conv10
        self.bn10 = orig_cnn16.bn10
        self.conv11 = orig_cnn16.conv11
        self.bn11 = orig_cnn16.bn11
        self.relu11 = orig_cnn16.relu11
        self.conv12 = orig_cnn16.conv12
        self.bn12 = orig_cnn16.bn12
        self.conv13 = orig_cnn16.conv13
        self.bn13 = orig_cnn16.bn13
        self.conv14 = orig_cnn16.conv14
        self.bn14 = orig_cnn16.bn14
        self.relu14 = orig_cnn16.relu14
        self.conv15 = orig_cnn16.conv15
        self.bn15 = orig_cnn16.bn15
        self.conv16 = orig_cnn16.conv16
        self.bn16 = orig_cnn16.bn16
        self.relu16 = orig_cnn16.relu16


    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.maxpool(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        conv_out.append(x)
        x = self.bn3(self.conv3(x))
        conv_out.append(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        conv_out.append(x)
        x = self.bn5(self.conv5(x))
        conv_out.append(x)
        x = self.relu6(self.bn6(self.conv6(x)))
        conv_out.append(x)
        x = self.bn7(self.conv7(x))
        conv_out.append(x)
        x = self.bn8(self.conv8(x))
        conv_out.append(x)
        x = self.relu9(self.bn9(self.conv9(x)))
        conv_out.append(x)
        x = self.bn10(self.conv10(x))
        conv_out.append(x)
        x = self.relu11(self.bn11(self.conv11(x)))
        conv_out.append(x)
        x = self.bn12(self.conv12(x))
        conv_out.append(x)
        x = self.bn13(self.conv13(x))
        conv_out.append(x)
        x = self.relu14(self.bn14(self.conv14(x)))
        conv_out.append(x)
        x = self.bn15(self.conv15(x))
        conv_out.append(x)
        x = self.relu16(self.bn16(self.conv16(x)))
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

class SelfONN18(nn.Module):
    def __init__(self, orig_selfonn18,dilate_scale=8):
        super(SelfONN18, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_selfonn18.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_selfonn18.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_selfonn18.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.conv1 = orig_selfonn18.conv1
        self.bn1 = orig_selfonn18.bn1
        self.relu1 = orig_selfonn18.relu1
        self.conv2 = orig_selfonn18.conv2
        self.bn2 = orig_selfonn18.bn2
        self.relu2 = orig_selfonn18.relu2
        self.conv3 = orig_selfonn18.conv3
        self.bn3 = orig_selfonn18.bn3
        self.relu3 = orig_selfonn18.relu3
        self.maxpool = orig_selfonn18.maxpool
        self.layer1 = orig_selfonn18.layer1
        self.layer2 = orig_selfonn18.layer2
        self.layer3 = orig_selfonn18.layer3
        self.layer4 = orig_selfonn18.layer4


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
        elif classname.find('SelfONN2dTransposed') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C2(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C2, self).__init__()
        self.use_softmax = use_softmax
        # SelfONN layer has q=2
        self.cbr = conv3x3_bn_self2(fc_dim, fc_dim // 4, 1)
        self.cbr2 = conv3x3_bn_self2(fc_dim // 4, fc_dim // 4, 1)
        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.cbr2(x)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C3(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C3, self).__init__()
        self.use_softmax = use_softmax
        # SelfONN layer has q=3
        self.cbr = conv3x3_bn_self3(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C5(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C5, self).__init__()
        self.use_softmax = use_softmax
        # SelfONN layer has q=5
        self.cbr = conv3x3_bn_self5(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x



# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.Tanh()
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.Tanh(),
            nn.Dropout2d(0.25),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)
    
class PPMDeepsupQ3(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.Tanh()
            ))
        self.ppm = nn.ModuleList(self.ppm)
        # SelfONN layer has q=3
        self.cbr_deepsup = conv3x3_bn_self3(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)
