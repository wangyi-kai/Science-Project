import torch
import torch.functional as F

from libs import *

class Net(nn.Module):
    def __init__(self, input, width, length, lb, ub):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input, width)
        self.hidden_layer = nn.ModuleList(nn.Linear(width, width) for i in range(length))
        self.output_layer = nn.Linear(width, 1)
        self.batch_norm1 = nn.BatchNorm1d(2)
        self.batch_norm = nn.BatchNorm1d(width)
        self.act = nn.Tanh()
        self.lb = torch.Tensor(lb)
        self.ub = torch.Tensor(ub)

    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.input_layer(x)
        x = torch.tanh(x)
        for i, li in enumerate(self.hidden_layer):
            x = li(x)
            #x = self.batch_norm(x)
            x = self.act(x)
        x = self.output_layer(x)
        return x

class Linear(nn.Module):
    def __init__(self, input, output):
        super(Linear, self).__init__()
        self.w = nn.Parameter(self.xavier_init(size=[input, output]))
        self.b = nn.Parameter(torch.zeros(1, output))

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        w = torch.empty(in_dim, out_dim)
        return torch.nn.init.xavier_normal_(w)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.w), self.b)
        #y = torch.tanh(y)
        return y

class Net2(nn.Module):
    def __init__(self, layers, lb, ub):
        super(Net2, self).__init__()
        self.layers = layers
        self.weights, self.bias = self.initial_NN(layers)
        self.lb = torch.Tensor([lb])
        self.ub = torch.Tensor([ub])

    def initial_NN(self, layers):
        weights = []
        bias = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = nn.Parameter(self.xavier_init(size=[layers[l], layers[l + 1]]))
            b = nn.Parameter(torch.zeros(layers[l + 1], 1))
            weights.append(W)
            bias.append(b)
        return weights, bias

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        w = torch.empty(in_dim, out_dim)
        return nn.init.xavier_normal_(w)

    def forward(self, X):
        num_layers = len(self.weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(1, num_layers - 2):
            W = self.weights[l]
            b = self.bias[l]
            H = F.linear(H, W, b)
        W = self.weights[-1]
        b = self.bias[-1]
        Y = F.linear(H, W, b)
        return Y

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# lb = [-1]
# ub = [1]
# layers = [2, 20, 20, 20, 20, 1]
# net = Net(2, 2, 4, lb, ub)
# print(net)
# for i, para in enumerate(net.parameters()):
#     print(i, para)

