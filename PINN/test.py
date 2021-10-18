import torch

from libs import *
from train import *
from net import *
from equation import *
from  data import *

x = torch.tensor([1.0], requires_grad=True)
y = x * x
u = torch.ones_like(x, requires_grad=True)
dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
#dyy = torch.autograd.grad(dy, u, grad_outputs=torch.ones_like(dy), create_graph=True)
print(dy)
#print(dyy)


