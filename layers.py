import numpy as np

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
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class QuadraticLayer(Layer):
    def __init__(self, input_size, output_size):
        self.quadWeights = np.random.rand(input_size, output_size, input_size)
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