import torch

from libs import *
from train import *
from net import *
from equation import *
from  data import *
net = torch.load('net_model2.pkl')

data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

input = torch.Tensor(X_star)
output = torch.Tensor(u_star)
print(input.shape, output.shape)
test_data = GetLoader(input, output)
test_dataset = DataLoader(test_data, batch_size=64)
# with torch.no_grad():
#     u_pred = net(input)
#     error_u = torch.norm(output - u_pred, 2) / torch.norm(output, 2)
#     print(output, u_pred)
#     error2 = (output - u_pred)
#     print('Error u: %e' % (error_u))



