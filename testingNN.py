from cnnClass import convolutionalNeuralNetwork
from qnnClass import quadraticNeuralNetwork
import numpy as np

third = np.random.randn(3, 2, 3) * .01 # j by i by j tensor
second = np.random.randn(2, 3) * .01 # i by j matrix for weight tensor
first = np.zeros(shape=(2)) # i by 1 vector for bias 'vector'
data = np.random.randn(3)

print(np.dot
    (data,
    np.dot(third, data)
    )
    + np.dot(second, data)
    + first
    )