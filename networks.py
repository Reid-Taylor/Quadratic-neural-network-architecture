import numpy as np
import time

from sklearn.feature_selection import f_regression

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    def add(self, layer):
        self.layers.append(layer)
        
    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        errorArray = {}
        tensorArray = {}
        checkemall = []
        delta = []
        for i in range(epochs):
            start_time = time.time()
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    (error, tensorw) = layer.backward_propagation(error, learning_rate)
                    if (tensorw.__class__ != str) : checkemall.append(tensorw)
            
            err /= samples
            errorArray[str(i)] = (err)
            print('Training: epoch %f/%f  |  error= %f' % (i+1, epochs, err))
            delta.append(time.time() - start_time)
            if (len(checkemall)):
                tensorArray[str(i)] = (np.count_nonzero(np.around(checkemall[-1], decimals=2)))
                print('Weight Tensor: %f' % (np.count_nonzero(np.around(checkemall[-1], decimals=2))))
            
        if (len(checkemall)):
            return (errorArray, tensorArray, tensorw, delta)
        return (errorArray, delta)