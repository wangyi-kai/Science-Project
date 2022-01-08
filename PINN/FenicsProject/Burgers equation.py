from __future__ import print_function

import scipy.io
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as iso
from mpl_toolkits import mplot3d
from dolfin import *

T = 1.0
num_steps = 100
dt = T / num_steps

nx = 255
nu = 0.01 / pi
x_left = -1.0
x_right = +1.0
dis = (x_right - x_left) / nx
mesh = IntervalMesh(nx, x_left, x_right)
V = FunctionSpace(mesh, 'CG', 2)

#define the boundary conditions
u_left = 0.0
def on_left(x, on_boundary):
    return (on_boundary and near(x[0], x_left))
bc_left = DirichletBC(V, u_left, on_left)

u_right = 0.0
def on_right ( x, on_boundary ):
    return ( on_boundary and near ( x[0], x_right ) )
bc_right = DirichletBC ( V, u_right, on_right )
bc = [bc_left, bc_right]

#define initial condition
u_init = Expression('-1 * sin(pi * x[0])', degree=1)
u_n = project(u_init, V, bc)
print(u_n.vector().get_local())
print(u_n.compute_vertex_values())

u = Function(V)
u_x = u.dx(0)
v = TestFunction(V)
v_x = v.dx(0)

dx = Measure("dx")
n = FacetNormal ( mesh )
F = u * v * dx - u_n * v * dx + dt * inner(u * u_x, v) * dx + dt * nu * inner(grad(u), grad(v)) * dx
#F = (dot(u - u_n, v) / dt + nu * inner(grad( u ), grad( v )) + inner( u * u.dx(0), v)) * dx
J = derivative(F, u)

t = 0
u_true = []
t_true = []
x_true = []
u_true.append(u_n.compute_vertex_values())
t_true.append(t)
for n in range(num_steps - 1):

    t += dt
    t_true.append(t)
    solve(F == 0, u, bc, J = J)

    vertex_values_u = u.compute_vertex_values()
    u_true.append(vertex_values_u)

    u_n.assign(u)

x_mesh = x_left
for i in range(nx + 1):
    x_true.append(x_mesh)
    x_mesh += dis
scipy.io.savemat('Burgers.mat', {'t': t_true, 'x': x_true, 'usol': u_true})

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(t_true, x_true, u_true)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# ax.set_title('Burgers Equation')
# plt.show()











