import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits import mplot3d
import math

data1 = scipy.io.loadmat('Burgers.mat')
t1 = data1['t'].flatten()[:, None]
x1 = data1['x'].flatten()[:, None]
Exact = np.real(data1['usol'])
u_star = Exact.flatten()[:, None]
#X_star = np.hstack((x, t))

data = scipy.io.loadmat('../burgers_shock.mat')
t = data['t'].flatten()[:, None]  # T x 1
x = data['x'].flatten()[:, None]  # N x 1
Exact2 = np.real(data['usol']).T  # T x N
# for i, j in enumerate(x1):
#     print(-math.sin(math.pi * x1[i]))
print((Exact - Exact2) / Exact2)
print(np.linalg.norm((Exact - Exact2) / Exact2, 2))

T, X = np.meshgrid(t, x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, Exact2.T)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
ax.set_title('Burgers Equation(True)')
plt.savefig('Burgers Equation(True).pdf')

T1, X1 = np.meshgrid(t1, x1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T1, X1, Exact.T)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
ax.set_title('Burgers Equation(Fenics)')
plt.savefig('Burgers Equation(Fenics).pdf')
plt.show()