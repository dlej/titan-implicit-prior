import torch
from torch import nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):

    def __init__(self,
                 n_features,
                 scale=1.0,
                 activation=F.relu,
                 dtype=torch.float):

        super().__init__()

        self.scale = scale
        self.activation = activation

        self.linear1 = nn.Linear(n_features, n_features, dtype=dtype)
        self.linear2 = nn.Linear(n_features, n_features, dtype=dtype)

    def forward(self, x):

        z = self.activation(self.linear1(x))
        w = self.linear2(z)

        return x + self.scale * w, z


class TITAN(nn.Module):

    def __init__(self,
                 in_features,
                 mid_features,
                 out_features,
                 depth=5,
                 resnet_activation=F.relu,
                 scale=1.0):
        super().__init__()

        self.linear_in = nn.Linear(in_features, mid_features)

        self.in_dd = nn.parameter.Parameter(torch.empty(mid_features))
        self.linears_dd = nn.ModuleList([
            nn.Linear(mid_features, mid_features, bias=False)
            for _ in range(depth - 1)
        ])
        self.linear_out = nn.Linear(mid_features, out_features, bias=False)

        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(mid_features,
                        scale=scale,
                        activation=resnet_activation) for _ in range(depth - 1)
        ])

        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(mid_features) for _ in range(depth - 1)])

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.in_dd.uniform_(-1, 1)

    def forward(self, input):

        x = self.linear_in(input)
        z = self.in_dd[(None, ) * (input.ndim - 1) + (..., )]

        for i, (l, rn, bn) in enumerate(
                zip(self.linears_dd, self.resnet_blocks, self.batchnorms)):
            _, r = rn(2 * (i + 1) * x)
            r = r / (len(self.linears_dd) + 1)
            z = F.relu(l(z) + r)
            s = z.shape
            z = z.reshape(-1, s[-1])
            z = bn(z).reshape(s)

        return torch.sigmoid(self.linear_out(z))
