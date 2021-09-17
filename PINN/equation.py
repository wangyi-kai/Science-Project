from libs import *

class AllenCahn():
    def __init__(self, net):
        super(AllenCahn, self).__init__()
        self.net = net

    def U_x(self, x):
        x = Variable(x, requires_grad=True)
        u = self.net(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u))[0][:, 1]
        return u_x

    def f(self, x):
        x = Variable(x, requires_grad=True)
        u = self.net(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dt = d[:, 0].reshape(-1, 1)
        dx = d[:, 1].reshape(-1, 1)
        dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0][:, 1].reshape(-1, 1)
        f = dt - 0.0001 * dxx + 5 * u[:, 0].reshape(-1, 1) ** 3 - 5 * u[:, 0].reshape(-1, 1)

    def sample(self, size):
        x = torch.cat((torch.rand(size, 1), 2 * torch.rand(size, 1)-1), dim=1)
        x_initial = torch.cat((torch.zeros(size, 1), torch.rand(size, 1)), dim=1)
        t_boundary = torch.rand(size, 1)
        left = torch.cat((t_boundary, -1 * torch.ones(size, 1)), dim=1)
        right = torch.cat((t_boundary, torch.ones(size, 1)), dim=1)
        t_x = torch.rand(size, 1)
        x_left = torch.cat((t_x, -1 * torch.ones(size, 1)), dim=1)
        x_right = torch.cat((t_x, torch.ones(size, 1)), dim=1)
        return x, x_initial, left, right, x_left, x_right

    def loss(self, size):
        x, x_initial, left, right, x_left, x_right = self.sample(size)
        f = self.f(x)
        physcis_error = torch.mean(torch.square(f))
        initial_error = torch.mean(torch.square(self.net(x_initial)[:,0] - x_initial[:, 1] ** 2 * torch.cos(np.pi * x_initial[:, 1])))
        boundary_error = torch.mean(torch.square(self.net(left)[:,0] - self.net(right)[:,0]))
        d_boundary_error = torch.mean(torch.square(self.U_x(x_left) - self.U_x(x_right)))
        error = physcis_error + initial_error + boundary_error + d_boundary_error
        return error

class Burgers():
    def __init__(self, net):
        self.net = net

    def f(self, x):
        x = Variable(x, requires_grad=True)
        u = self.net(x)
        d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dt = d[:,1]
        dx = d[:,0]
        dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0][:,0]
        f = dt + u * dx - (0.01 / np.pi) * dxx
        return f

    def loss(self, x, x_initial, left, right):
        x = torch.Tensor(x)
        x_initial = torch.Tensor(x_initial)
        left = torch.Tensor(left)
        right = torch.Tensor(right)
        f = self.f(x)
        physics_error = torch.mean(torch.square(f))
        initial_error = torch.mean(torch.square(self.net(x_initial)[:,0] + torch.sin(np.pi * x_initial[:,0])))
        boundary_error = torch.mean(torch.square(self.net(left)) + torch.square(self.net(right)))
        loss = physics_error + initial_error + boundary_error
        return loss

    def loss1(self, x_f, x_u, u):
        x_f = torch.Tensor(x_f)
        x_u = torch.Tensor(x_u)
        u = torch.Tensor(u)
        f = self.f(x_f)
        b_error = self.net(x_u) - u
        loss = torch.mean(torch.square(f)) + torch.mean(torch.square(b_error))
        return loss








