from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

#parameters
nx = 100
x_left = -1.0
x_right = 1.0
T = 1
time_step = 100
dt = T / time_step
order = 2
#define mesh
mesh = IntervalMesh(nx, x_left, x_right)
V = FunctionSpace(mesh, 'P', order)
u = Function(V)
v = TestFunctions(V)

#initial condition
u_init = Expression('pow(x[0], 2) * cos(pi * x[0])', degree = 2)
u_n = interpolate(u_init, V)

#boundary condition
def on_left(x, on_boundary):
    return (on_boundary and near(x[0], x_left))
bc_left = DirichletBC(V, on_left)

def on_right(x, on_boundary):
    return (on_boundary and near(x[0], x_right))

l1 = 0.0001
l2 = 5.0
dx = Measure("dx")
#variation formula
F = u * v * dx - u_n * v * dx + dt * l1 * dot(grad(u), grad(v)) * dx + \
    dt * l2 * (u ** 3 - u) * v * dx
J = derivative(F, u)


