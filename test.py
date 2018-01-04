import scipy as sp
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from  sklearn.utils import shuffle
from loadData import loadTestData
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

X_test, Y_test = loadTestData();
X_test, Y_test = shuffle(X_test, Y_test)


network = input_data(shape=[None, 32, 32, 3])
network = conv_2d (
    network,
    16,
    5,
    activation='relu',
    weights_init='xavier')

network = batch_normalization(network)

network = conv_2d (
    network,
    16,
    5,
    activation='relu',
    weights_init='xavier')

network = max_pool_2d( network, 2 )
network = batch_normalization(network)

network = conv_2d (
    network,
    32,
    3,
    activation='relu',
    weights_init='xavier')

network = batch_normalization(network)

network = conv_2d (
    network,
    32,
    3,
    activation='relu',
    weights_init='xavier')

network = max_pool_2d( network, 2 )

network = batch_normalization(network)


network = fully_connected(network, 512, activation='relu', weights_init='xavier')
network = dropout( network, 0.4)


network = fully_connected(network, 10, activation='softmax', weights_init='xavier')
#print(output.shape)


network = regression( network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("svhn_model.tfl")

total_samples = len(X_test)
correct_predict = 0.
for i in xrange(len(X_test)):
    prediction = model.predict([X_test[i]])
    digit = np.argmax(prediction)
    label = np.argmax(Y_test[i])
    if(digit == label):
        correct_predict += 1

print(correct_predict/total_samples)