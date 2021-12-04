import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class convolutionalNeuralNetwork:
    def __init__(self, dims, learning_rate=0.05, iterations=1000):
        self.layer_dims = dims
        self.parameters = self.initialize_parameters_deep(self.layer_dims)
        self.learning_rate = learning_rate
        self.num_iterations = iterations

    def initialize_parameters_deep(layer_dims): #vector
        parameters = {}
        L = len(layer_dims) #number of layers
        for l in range(1,L):
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * .01 # i by j matrix for weight matrix
            parameters['b'+str(l)] = np.zeros(shape=(layer_dims[l],1)) # i by 1 vector for bias 'vector'
            assert(parameters['W'+str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b'+str(l)].shape == (layer_dims[l], 1))

        return parameters
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def linear_model_forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2 # number of layers in the neural network, floor division by 2 because each layer has a weights and a bias entry in {parameters}
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev,
                self.parameters["W"+str(l)],
                self.parameters["b"+str(l)],
                activation = 'relu'
                )
            caches.append(cache)
        AL, cache = self.linear_activation_forward(A,
            self.parameters['W'+str(L)],
            self.parameters['b'+str(L)],
            activation='sigmoid'
            )
        caches.append(cache)
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(AL)) + np.multiply((1-Y),np.log(1-AL))) #cross entropy loss function
        cost = np.squeeze(cost)
        assert(cost.shape == ()) #scalar, no dimensionality
        return cost

    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, cache[0].T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[1].T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation=="relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation=="sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL ))

        current_cache = caches[-1]
        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_backward(self.sigmoid_backward(dAL, current_cache[1]), current_cache[0])

        for l in reversed(range(L-1)):
            current_cache = caches[1]
            dA_prev_temp, dW_temp, db_temp = self.linear_backward(self.sigmoid_backward(dAL, current_cache[1]), current_cache[0])
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        
        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
            self.parameters["b" + str(l+1)] = self.parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    
    def train(self, X, Y):
        costs = []
        for i in range(0, self.num_iterations):
            AL, caches = self.linear_model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads, self.learning_rate)
            if i % 100 == 0:
                costs.append(cost)
                if self.print_costs:
                    print("Cost after iteration %i: %f" %(i, cost))
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate =' + str(self.learning_rate))
        plt.show()
    
    def sigmoid(Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(Z):
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache

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
        
        assert (dZ.shape == Z.shape)    
        return dZ