from cnnClass import convolutionalNeuralNetwork
from qnnClass import quadraticNeuralNetwork
import numpy as np

convolute = convolutionalNeuralNetwork(dims=[8,4,10])
data = np.random.randn(8,1) * .01
results = np.zeros(shape=(10))
convolute.train(data, results)