
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmLoss(nn.CrossEntropyLoss):
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, aux=False, aux_weight=0.4, weight=None, size_average=True, ignore_index=-1):
        super(SegmLoss, self).__init__(weight, size_average, ignore_index)

        self.se_loss =se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmLoss, self).forward(*inputs)

        elif not self.se_loss:
            (pred1, pred2), target = tuple(inputs)
            loss1 = super(SegmLoss, self).forward(pred1, target)
            loss2 = super(SegmLoss, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2

        elif not self.aux:
            (pred, se_pred), target = tuple(inputs)
            se_target = self._get_batch_label_vector(
                target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2

        else:
            (pred1, se_pred, pred2), target = tuple(inputs)
            se_target = self._get_batch_label_vector(
                target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmLoss, self).forward(pred1, target)
            loss2 = super(SegmLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3


    @staticmethod
    def _get_batch_label_vector(target, nclass):
            # target is a 3D Variable BxHxW, output is 2D BxnClass
            batch = target.size(0)
            tvect = torch.zeros(batch, nclass)
            for i in range(batch):
                hist = torch.histc(target[i].cpu().data.float(),
                                bins=nclass, min=0,
                                max=nclass-1)
                vect = hist > 0
                tvect[i] = vect
            return tvect
