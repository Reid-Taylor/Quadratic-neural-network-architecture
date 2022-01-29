from typing import DefaultDict
import numpy as np
from numpy.matlib import eye
from numpy.linalg import inv

class Layer: #interface with which to define layers
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propogation(self,input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1,output_size))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return (input_error, self.weights)

class QuadraticLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.quadWeights = np.random.rand(input_size, output_size, input_size) - 0.5
        self.bias = np.zeros((1,output_size))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.tensordot(np.squeeze(np.tensordot(self.input, self.quadWeights, axes=1)),self.input.T,axes=1).T + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        e = np.matlib.eye(n=self.output_size, M=self.input_size)
        if (self.output_size > self.input_size):
            e[(self.input_size):(self.output_size-1),:] = (1/self.input_size)
        elif (self.output_size < self.input_size):
            e[:,(self.output_size):(self.input_size-1)] = (1/self.output_size)
        else: 
            e = np.identity(self.output_size)
        try:
            input_error = np.dot(output_error, np.squeeze(np.tensordot(np.dot(output_error, e), self.quadWeights,axes=(1,0))))
        except:
            print('exception')
            input_error = np.dot(output_error, np.expand_dims(np.squeeze(np.tensordot(np.dot(output_error, e), self.quadWeights,axes=(1,0))),axis=0))
        quad_weights_error = np.tensordot(self.input.T, np.expand_dims(np.dot(output_error.T,self.input), axis=0), axes=(1,0))
        self.quadWeights -= learning_rate * quad_weights_error
        self.bias -= learning_rate * output_error
        return (input_error, self.quadWeights)

class ExponentialLayer(Layer):
    def __init__(self, input_size, output_size):
        self.quadWeights = np.random.rand(input_size, output_size, input_size) - 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1,output_size))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.tensordot(np.squeeze(np.tensordot(self.input, self.quadWeights, axes=1)),self.input.T,axes=1).T + np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.tensordot(self.input.T, output_error, axes=1)
        quad_weights_error = np.tensordot(np.expand_dims(np.tensordot(self.input.T,output_error,axes=1),axis=2),self.input,axes=1)
        self.quadWeights -= learning_rate * quad_weights_error
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class PowerLayer(Layer):
    def __init__(self, input_size, output_size):
        self.powerWeights = np.diag(np.diag(np.random.rand(input_size, input_size)))
        self.powerBase = np.random.rand((1,output_size), (1/output_size))
        self.power = np.random.random()
        self.bias = np.zeros((1,output_size))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.diag(np.exp(np.abs(self.input * np.log(self.powerWeights)))) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error ** self.power
        input_error = np.diag(np.dot(np.diag(np.log(output_error)), np.linalg.inv(np.log(self.powerWeights))))

        power_error = np.dot(self.input.T, output_error)

        self.power -= learning_rate * power_error
        self.bias -= learning_rate * output_error
        return input_error

class SinLayer(Layer):
    def __init__(self, input_size, output_size):
        self.powerWeights = np.random.rand(input_size, output_size, input_size) - 0.5
        self.powerBase = np.full((1,output_size), (1/output_size))
        self.bias = np.zeros((1,output_size))
    
    def forward_propagation(self, input_data):
        self.input = input_data
        # self.output = 
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        # input_error = np.dot(output_error, self.weights.T)
        #for powerWeightsError consider the laGrange Error Bound...
        self.bias -= learning_rate * output_error
        # return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return (self.activation_prime(self.input) * output_error , '')
