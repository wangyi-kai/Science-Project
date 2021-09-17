import os

import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs import *
from train import *
from net import *
from equation import *
from data import *
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
lr = 0.001
N_f = 4000
N_u = 200

data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]
lb = X_star.min(0)
ub = X_star.max(0)

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

epoch = 50
LBFGS_error = []
net = Net(2, 20, 8, lb, ub)
net.apply(weight_init)
equation = Burgers(net)
optimizer = optim.LBFGS(net.parameters(),line_search_fn='strong_wolfe')
start = time.time()
for e in range(epoch):
    def closure():
        optimizer.zero_grad()
        loss = equation.loss1(X_f_train, X_u_train, u_train)
        loss.backward()
        return loss
    optimizer.step(closure)
    loss = closure()
    end = time.time() - start
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("Epoch {} - lr {} -  loss: {} - time:{}".format(e, lr, loss, end))
    LBFGS_error.append(loss.item())

#save the model
#plt.plot(LBFGS_error)
#plt.show()
np.savetxt('LBFGS_training_error.csv', LBFGS_error)
torch.save(net, 'LBFGS_net_model.pkl')

epoch2 = 2000
Adam_error = []
# net = Net(2, 20, 8, lb, ub)
# net.apply(weight_init)
# equation = Burgers(net)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000)
optimizer = optim.Adam(net.parameters(),0.001)
start_time = time.time()
for e in range(epoch2):
    optimizer.zero_grad()
    loss = equation.loss1(X_f_train, X_u_train, u_train)
    loss.backward()
    optimizer.step()
    end = time.time() - start
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("Epoch {} - lr {} -  loss: {} - time:{}".format(e, lr, loss, end))
    Adam_error.append(loss.item())

input = torch.Tensor(X_star)
output = torch.Tensor(u_star)
test_data = GetLoader(input, output)
test_dataset = DataLoader(test_data, batch_size=1024, shuffle=True)
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        x_test = data[0]
        label = data[1]
        u_pred = net(x_test)
        error_u = torch.norm(label - u_pred, 2) / torch.norm(label, 2)
        print('Error u: %e' % (error_u))