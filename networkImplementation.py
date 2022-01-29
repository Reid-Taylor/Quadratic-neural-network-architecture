from networks import Network
from layers import FCLayer, QuadraticLayer, ActivationLayer
from activations import sigmoid, sigmoid_prime, tanh, tanh_prime, mse, mse_prime
import numpy as np
import csv

AGES = 48

# This is for loading in MNIST data for image recognition
# package imports
from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# create regression data
# SAMPLES = 10000
# # ------------- Generate Dataset ------------------------
# X = np.reshape(np.linspace(0, 1, SAMPLES),(SAMPLES, 1))
# Y = X**2
# # ----------- split dataset into training set and test set -------------------------------------------
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=.85)
for j in range(AGES):
    # Network
    net = Network()
    # net.add(FCLayer(1, 5))
    # net.add(ActivationLayer(tanh, tanh_prime))
    # net.add(QuadraticLayer(784, 10))
    net.add(FCLayer(784, 10))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # train on 1000 samples
    # as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
    net.set_loss(mse, mse_prime)
    (errorArray, tensorArray, tensorw, delta) = net.fit(x_train[0:1000], y_train[0:1000], epochs=50, learning_rate=0.15)

    ind = list(tensorArray.keys())
    nerrorArray = list(errorArray.values())
    nTensorArray = list(tensorArray.values())
    # data = [['attempt','epoch','training_error','nonzero_params','delta_time']]
    data = []
    for i in range(len(ind)):
        data.append([str(j+48),str(ind[i]), str(nerrorArray[i]), str(nTensorArray[i]), str(delta[i])])

    with open('epochs.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
