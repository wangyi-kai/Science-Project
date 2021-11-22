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
pi = 3.14159
nu = 0.01 / pi
x_left = -1.0
x_right = 1.0
dis = (x_right - x_left) / nx
mesh = IntervalMesh(nx, x_left, x_right)
V = FunctionSpace(mesh, 'P', 1)

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
u_init = Expression("-1 * sin(pi * x[0])", degree=1, pi = pi)
u_n = interpolate(u_init, V)

u = Function(V)
u_x = u.dx(0)
f = Constant(0)
v = TestFunction(V)
v_x = v.dx(0)

dx = Measure("dx")
n = FacetNormal ( mesh )
#F = dot(u - u_n, v) * dx + dt * inner(u * u_x, v) * dx + dt * nu * inner(grad(u), grad(v)) * dx
F = \
  ( \
    dot ( u - u_n, v ) / dt \
  + nu * inner ( grad ( u ), grad ( v ) ) \
  + inner ( u * u.dx(0), v ) \
  - dot ( f, v ) \
  ) * dx

J = derivative(F, u)

t = 0
u_true = []
u_d = []
t_true = []
x_true = []

for n in range(num_steps):
    # for i in range(nx + 1):
    #     t_true.append(t)
    t_true.append(t)

    solve(F == 0, u, bc, J = J)
    t += dt
    vertex_values_u = u.compute_vertex_values()
    u_true.append(vertex_values_u)
    # for i in range(len(vertex_values_u)):
    #     u_true.append(vertex_values_u[i])
    u_n.assign(u)
x_mesh = x_left
for i in range(nx + 1):
    x_true.append(x_mesh)
    x_mesh = x_mesh + dis
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(t_true, x_true, u_true)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# ax.set_title('Burgers Equation')
# plt.show()
scipy.io.savemat('Burgers.mat', {'t' : t_true, 'x' : x_true, 'usol' : u_true})
# print(t_true)
# print(x_true)
# print(u_true)
# print(len(t_true))
# print(len(x_true))
# print(len(u_true))
#np.savetxt('FenicsProject/generate_data.csv', u_true)










