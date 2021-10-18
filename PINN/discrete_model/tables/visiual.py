import numpy as np
import matplotlib.pyplot as plt

pytorch_error = np.loadtxt('training_error_pytorch.csv')
tensorflow_error = np.loadtxt('training_error_tensorflow.csv')

plt.plot(pytorch_error, label = 'pytorch')
#plt.plot(tensorflow_error[:5000:100], label = 'tensorflow')
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Training Error')
plt.legend()
plt.savefig('discrete_model/figures/error_pytorch.pdf')
plt.show()