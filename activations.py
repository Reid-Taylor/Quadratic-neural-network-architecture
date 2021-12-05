import numpy as np

def mse(y_true, y_hat):
    return np.mean(np.power(y_true-y_hat, 2))

def mse_prime(y_true, y_hat):
    return 2*(y_hat-y_true)/y_true.size

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    x[x>=0] = 1
    x[x<0] = 0
    return x

def identity(x):
    return x

def identity_prime(x):
    return 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):        
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid * (1-sigmoid)