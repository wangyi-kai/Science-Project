from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as iso

from dolfin import *

T = 1.0
num_steps = 10
dt = T / num_steps

nx = 4
pi = 3.14159
nu = 0.01 / pi
x_left = -1.0
x_right = 1.0
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
#a, L = lhs(F), rhs(F)
J = derivative(F, u)

t = 0
# k = 0
# t_plot = 0.0
# t_final = 1.0
# while (True):
#     if (k % 10 == 0):
#       plot ( u, title = ( 'burgers time viscous %g' % ( t ) ) )
#       plt.grid ( True )
#       plt.show()
#       filename = ( 'burgers_time_viscous_%d.png' % ( k ) )
#       plt.savefig ( filename )
#       print ( 'Graphics saved as "%s"' % ( filename ) )
#       plt.close ( )
#       t_plot = t_plot + 0.1
#
#     if ( t_final <= t ):
#       print ( '' )
#       print ( 'Reached final time.' )
#       break
#
#     k += 1
#     t += dt
#     solve ( F == 0, u, bc, J = J)
#     print(t)
#     vertex_values_u = u.compute_vertex_values()
#     print(vertex_values_u)
#     u_n.assign(u)
u_true = []
t_true = []
x_true = []
for n in range(num_steps):
    for i in range(nx):
        t_true.append(t)
    #u_D.t = t
    solve(F == 0, u, bc, J = J)
    #plot(u)
    t += dt
    #print(mesh)
    vertex_values_u = u.compute_vertex_values()
    u_true.append(vertex_values_u.squeeze())
    u_n.assign(u)
#plot(mesh)
print(t_true)
plt.show()
#print(u_true)
np.savetxt('generate_data.csv', u_true)










