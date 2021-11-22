import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

data = scipy.io.loadmat('Burgers.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
u_star = Exact.flatten()[:, None]
#X_star = np.hstack((x, t))
print(Exact)
data = scipy.io.loadmat('/home/wangyikai/Documents/Research/Example/PINN/burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T
print(Exact)