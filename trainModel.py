import scipy as sp
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from  sklearn.utils import shuffle
from loadData import loadTrainingData
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

X_train, Y_train = loadTrainingData();
X_train, Y_train = shuffle(X_train, Y_train)
X_val = X_train[0:7000]
Y_val = Y_train[0:7000]
X_train = X_train[7000:]
Y_train = Y_train[7000:]



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

network = regression( network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X_train, Y_train, n_epoch=10, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=100,
          snapshot_epoch=True,
          run_id='svhn_model')

model.save("svhn_model.tfl")









