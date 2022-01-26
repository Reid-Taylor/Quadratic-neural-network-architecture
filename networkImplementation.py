from networks import Network
from layers import FCLayer, QuadraticLayer, ActivationLayer
from activations import sigmoid, sigmoid_prime, tanh, tanh_prime, mse, mse_prime
import numpy as np

'''
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
'''
# create regression data
SAMPLES = 10000
# ------------- Generate Dataset ------------------------
X = np.reshape(np.linspace(0, 1, SAMPLES),(SAMPLES, 1))
Y = X**2
# ----------- split dataset into training set and test set -------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=.85)

# Network
net = Network()
net.add(FCLayer(1, 5))                # input_shape=(1, 1)    ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(QuadraticLayer(5, 2))                    # input_shape=(1, 50)      ;   output_shape=(1, 1)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# net.add(FCLayer(50, 1))                    # input_shape=(1, 50)      ;   output_shape=(1, 1)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.set_loss(mse, mse_prime)
errorArray = net.fit(x_train[0:1000], y_train[0:1000], epochs=50, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
import matplotlib.pyplot as plt
plt.plot(errorArray.keys(), errorArray.values())
plt.title('Neural Network -- Constant Function')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim([0,51])
plt.ylim([0,1])
plt.xticks(np.arange(0, 51, step=5))
plt.figtext(0.5, 0.15, f"Final Error: {round(list(errorArray.values())[-1],6)*100}%", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# plt.savefig('NN-constant.png', dpi=400)
plt.show()