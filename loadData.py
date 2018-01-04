import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

trainingData = sio.loadmat('/Users/arungta/Downloads/Data/train_32x32.mat');
testData = sio.loadmat('/Users/arungta/Downloads/Data/test_32x32.mat');

def loadTrainingData():
	X = np.asarray(trainingData['X'])
	X_train = []
	for i in xrange(X.shape[3]) :
		X_train.append(X[:,:,:,i])
	X_train = np.asarray(X_train)

	Y = np.asarray(trainingData['y'])
	Y_train = []
	for i in xrange(Y.size) :
		if(Y[i]%10 == 0) :
			Y[i] = 0;
		Y_train.append(Y[i])
	Y_train = np.asarray(Y_train)
	onehot_encoder = OneHotEncoder(sparse=False)
	Y_train = onehot_encoder.fit_transform(Y_train)
	return X_train, Y_train


def loadTestData():
	X = np.asarray(testData['X'])
	X_test = []
	for i in xrange(X.shape[3]) :
		X_test.append(X[:,:,:,i])
	X_test = np.asarray(X_test)

	Y = np.asarray(testData['y'])
	Y_test = []
	for i in xrange(Y.size) :
		if(Y[i]%10 == 0) :
			Y[i] = 0;
		Y_test.append(Y[i])
	Y_test = np.asarray(Y_test)
	onehot_encoder = OneHotEncoder(sparse=False)
	Y_test = onehot_encoder.fit_transform(Y_test)
	return X_test, Y_test


