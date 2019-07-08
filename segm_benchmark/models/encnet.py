

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment import SegmNet
from .fcn import FCNHead
from segm_benchmark.layers import encoding
from segm_benchmark.layers.loss import SegmLoss

class EncNet(SegmNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        
        self.head = EncHead([512, 1024, 2048], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'], lateral=kwargs['lateral'], norm_layer=norm_layer, up_kwargs=self.up_kwargs)
        
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

        self.loss = SegmLoss(se_loss=se_loss, aux=aux, nclass=nclass)

    def forward(self, x, targets=None):
        imsize = x.size()[2:]
        features = self.bone_forward(x)

        x = list(self.head(*features))
        x[0] = F.upsample(x[0], imsize, **self.up_kwargs)

        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self.up_kwargs)
            x.append(auxout)

        if targets is not None:
            return dict(total_loss=self.loss(tuple(x), targets))
        else:
            return tuple(x)

        
class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False, norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], 512, 1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True)
        ) if jpu else \
            nn.Sequential(
                nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            )
        
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)
                )
            ])

            self.fusion = nn.Sequential(
                nn.Conv2d(512 * 3, 512, kernel_size=3, padding=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            )
        
        self.encmodule = EncModule(512, out_channels, ncodes=32, se_loss=se_loss, norm_layer=norm_layer)
        
        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(512, out_channels, 1)
        )

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])

        return tuple(outs)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss =se_loss

        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            encoding.Encoding(D=in_channels, K=ncodes),
            nn.SyncBatchNorm(ncodes),
            nn.ReLU(inplace=True),
            encoding.Mean(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))

        return tuple(outputs)
