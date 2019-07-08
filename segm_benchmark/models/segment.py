
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from segm_benchmark.layers import JPU
from segm_benchmark.models import resnet
from segm_benchmark.utils.metrics import batch_intersection_union, batch_pix_accuracy

_up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class SegmNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, jpu=True, dilated=False, norm_layer=None, base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], **kwargs):
        super(SegmNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size

        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated, norm_layer=norm_layer)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated, norm_layer=norm_layer)
        else:
            raise NotImplementedError("Unknown backbone: {}".format(backbone))

        self.up_kwargs = _up_kwargs
        self.backbone = backbone
        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, up_kwargs=_up_kwargs) if jpu else None

    def bone_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        
        correct, labeled = batch_pix_accuracy(pred, target)
        inter, union = batch_intersection_union(pred, target, self.nclass)
        return correct, labeled, inter, union