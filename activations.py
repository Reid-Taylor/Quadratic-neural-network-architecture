import numpy as np

def mse(y_true, y_hat):
    return np.mean(np.power(y_true-y_hat, 2))

def mse_prime(y_true, y_hat):
    return 2*(y_hat-y_true)/y_true.size

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
        
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ