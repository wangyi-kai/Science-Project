import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class FeedForwardSubNet(nn.Module):
    def __init__(self, config):
        self.input = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width)  for i in range(length))
        self.output = nn.Linear(width, out_dim)
        self.act = nn.ReLU()
        self.batch1 = nn.BatchNorm1d(in_dim)
        self.batch2 = nn.BatchNorm1d(width)
        self.batch3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.batch1(x)
        x = self.input(x)
        x = self.act(x)
        for i, li in enumerate(self.hidden):
            x = li(x)
            x = self.batch2(x)
            x = nn.ReLU(x)
        x = self.output(x)
        x = self.batch3(x)
        return x

class Model(nn.Module):
    def __init__(self, config, bsde):
        super(Model, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = Variable(torch.rand([1]))
        self.z_init = Variable(torch.rand(1, self.eqn_config.dim))
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval-1)]

    def forward(self, inputs):
        dw, x = inputs
        time_stamp = np.range(0, self.eqn_config.num_time_intervals) * self.bsde.delta_t
        all_one_vec = torch.ones(shape=torch.stack([dw.shape[0], 1]), dtype=self.net_config.dtype)
        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, self.z_init)

        for t in range(0, self.bsde.num_time_interval - 1):
            y = y - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:,:, t], y, z)) \
                + torch.sum(z * dw[:,:,t], 1, keepdim=True)
            z =
