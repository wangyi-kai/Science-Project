import torch
import torch.nn as nn
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self, input, width, length):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input, width)
        self.hidden_layer = nn.ModuleList(nn.Linear(width, width) for i in range(length))
        self.output_layer = nn.Linear(width, 1)
        self.act = nn.Tanh()
        #self.lb = torch.Tensor(lb)
        #self.ub = torch.Tensor(ub)

    def forward(self, x):
        #x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.input_layer(x)
        x = torch.tanh(x)
        for i, li in enumerate(self.hidden_layer):
            x = li(x)
            x = self.act(x)
        x = self.output_layer(x)
        return x

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class PhysicsInformedNN():
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.x0 = torch.tensor(x0, requires_grad=True).float().to(device)
        self.x1 = torch.tensor(x1, requires_grad=True).float().to(device)

        self.u0 = torch.tensor(u0).float().to(device)

        self.layers = layers
        self.dt = torch.tensor(dt).float().to(device)
        self.q = max(q, 1)
        self.dummy = torch.ones(self.x0.shape[0], self.q, requires_grad=True)

        self.dnn = DNN(layers).to(device)
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.iter = 0
        self.loss_table = []

        # Load IRK weights
        tmp = np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2)
        self.IRK_weights = torch.tensor(np.reshape(tmp[0:q ** 2 + q], (q + 1, q))).float().to(device)
        self.IRK_times = torch.tensor(tmp[q ** 2 + q:]).float().to(device)

    def fwd(self, U, x):
        g = torch.autograd.grad(U, x,
                                grad_outputs=self.dummy,
                                retain_graph=True,
                                create_graph=True)[0]
        return torch.autograd.grad(g, self.dummy,
                                   grad_outputs=torch.ones_like(g),
                                   retain_graph=True,
                                   create_graph=True)[0]

    def net_u(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        u = self.dnn(x)
        return u

    def net_U0(self, x):
        """ The pytorch autograd version of calculating residual """
        u1 = self.net_u(x)
        u = u1[:,:-1]
        nu = 0.01 / np.pi
        # u_x = torch.autograd.grad(
        #     u, x,
        #     grad_outputs=torch.ones_like(u),
        #     retain_graph=True,
        #     create_graph=True,
        #     only_inputs=False
        # )[0]
        # u_xx = torch.autograd.grad(
        #     u_x, x,
        #     grad_outputs=torch.ones_like(u_x),
        #     retain_graph=True,
        #     create_graph=True,
        #     only_inputs=False
        # )[0]
        u_x = self.fwd(u, x)
        u_xx = self.fwd(u_x, x)
        f = -u * u_x + nu * u_xx
        U0 = u1 - self.dt * torch.matmul(f, self.IRK_weights.T)
        return U0

    def loss_func(self):
        self.optimizer.zero_grad()

        U0_pred = self.net_U0(self.x0)
        U1_pred = self.net_u(self.x1)
        loss_u = torch.mean((self.u0 - U0_pred) ** 2)
        loss_u1 = torch.mean(U1_pred ** 2)

        loss = loss_u + loss_u1
        loss.backward()
        self.loss_table.append(loss.item())
        self.iter += 1

        if self.iter % 1 == 0:
            print('Iter %d, Loss %.5e' % (self.iter, loss.item()))

        return loss

    def loss_func2(self):
        #self.optimizer_Adam.zero_grad()
        U0_pred = self.net_U0(self.x0)
        U1_pred = self.net_u(self.x1)
        loss_u = torch.mean((self.u0 - U0_pred) ** 2)
        loss_u1 = torch.mean(U1_pred ** 2)

        loss = loss_u + loss_u1
        #loss.backward()
        return loss

    def train(self, epoch):
        start_time = time.time()
        for it in range(epoch):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_func2()
            loss.backward()
            self.optimizer_Adam.step()
            end = time.time() - start_time
            self.loss_table.append(loss.item())
            if it % 10 == 0:
                print('Iter %d, Loss %.5e' % (it, loss.item()))

        self.dnn.train()
        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, x):
        x = torch.tensor(x, requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x)
        u = u.detach().cpu().numpy()
        return u



def main_loop(q, skip, num_layers, num_neurons):
    layers = np.concatenate([[1], num_neurons * np.ones(num_layers), [q + 1]]).astype(int).tolist()

    lb = np.array([-1.0])
    ub = np.array([1.0])

    N = 250

    data = scipy.io.loadmat('burgers_shock.mat')

    t = data['t'].flatten()[:, None]  # T x 1
    x = data['x'].flatten()[:, None]  # N x 1
    Exact = np.real(data['usol']).T  # T x N

    idx_t0 = 10
    idx_t1 = idx_t0 + skip
    dt = t[idx_t1] - t[idx_t0]

    # Initial data
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    # Boudanry data
    x1 = np.vstack((lb, ub))

    # Test data
    x_star = x

    model = PhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q)
    model.train(10000)

    U1_pred = model.predict(x_star)

    error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)

    return error


if __name__ == "__main__":

    q = [1, 2, 4, 8, 16, 32, 64, 100, 500]
    skip = [20, 40, 60, 80]

    num_layers = [1, 2, 3]
    num_neurons = [10, 25, 50]

    error_table_1 = np.zeros((len(q), len(skip)))
    error_table_2 = np.zeros((len(num_layers), len(num_neurons)))

    for i in range(len(q)):
        for j in range(len(skip)):
            error_table_1[i, j] = main_loop(q[i], skip[j], num_layers[-1], num_neurons[-1])

    for i in range(len(num_layers)):
        for j in range(len(num_neurons)):
            error_table_2[i, j] = main_loop(q[-1], skip[-1], num_layers[i], num_neurons[j])

    np.savetxt('discrete_model/tables/error_table_1.csv', error_table_1, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')
    np.savetxt('discrete_model/tables/error_table_2.csv', error_table_2, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')