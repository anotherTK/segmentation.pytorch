
import torch
import torch.nn as nn
import torch.nn.functional as F

from segm_benchmark.utils.encoding import scaled_l2, aggregate

class Encoding(nn.Module):
    r"""
    Encoding Layer: a learnable residual encoder.
    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    """

    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        _std = 1.0 / ((self.K * self.D) ** (1.0 / 2))
        self.codewords.data.uniform_(-_std, _std)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # B, D, N -> B, N, D
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # B,D,H,W -> B,HW, D
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError("Encoding Layer unknown input dims!")
        # assignment weights BxNxK
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)

        return E

    def __repr__(self):
        return self.__class__.__name__ + '(N x ' + str(self.D) + '=>' + str(self.K) + ' x ' + str(self.D) + ')'


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)
