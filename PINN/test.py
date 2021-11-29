from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as iso
from mpl_toolkits import mplot3d
from dolfin import *

num = 10
mesh = IntervalMesh(num, -1, 1)
V = FunctionSpace(mesh, 'P', 1)
u_init = Expression("x[0] + 0.5", degree=1)

u_n = interpolate(u_init, V)
u_value = u_n.vector().get_local()
u_node = u_n.compute_vertex_values()
print(u_n)
print(u_value)
print(u_node)


