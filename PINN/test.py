import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

pi = 3.14159265
N_f = 1000
N_u = 200
data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])

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
print(x)
X_u_train = np.vstack([xx1, xx2, xx3])
#X_f_train = lb + (ub - lb) * lhs(2, N_f)
#X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]


