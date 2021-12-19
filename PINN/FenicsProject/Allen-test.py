import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits import mplot3d
import math

data = scipy.io.loadmat('Allen-Cahn.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
#u_star = Exact.flatten()[:, None]

data1 = scipy.io.loadmat('../AC.mat')
t1 = data1['tt'].flatten()[:, None]
x1 = data1['x'].flatten()[:, None]
Exact1 = np.real(data1['uu'])

print(Exact.T - Exact1)

T, X = np.meshgrid(t, x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, Exact.T)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
ax.set_title('Allen-Cahn(Fenics)')
#plt.savefig('Burgers Equation(True).pdf')

T1, X1 = np.meshgrid(t1, x1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T1, X1, Exact1)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
ax.set_title('Allen-Cahn(True)')
plt.show()