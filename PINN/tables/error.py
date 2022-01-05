import matplotlib.pyplot as plt
import numpy as np
pytorch_error = np.loadtxt('training_error_pytorch.csv')
tensorflow_error = np.loadtxt('training_error_tensorflow.csv')

plt.plot(pytorch_error, label = 'pytorch')
plt.plot(tensorflow_error, label = 'tensorflow')
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Training Error')
plt.legend()
plt.savefig('error_compare.pdf')
plt.show()