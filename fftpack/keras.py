import scipy.fftpack as fftpack
import numpy as np
import pandas as pd
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Embedding, LSTM
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split

tt = pd.read_csv("tones_noise.txt", header = None)
X = tt.values[:, 0:501]
y = tt.values[:, 501]

XF = np.abs(fftpack.fft(X))


"""
max_features = 1024
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=)
"""

c_intervals = np.linspace(1, 10, num = 3)
gamma_intervals = np.linspace(0.1, 10, num = 3)

#cross-validation

num_folds = 3
X_train_folds = np.array(np.array_split(X, num_folds))
XF_train_folds = np.array(np.array_split(XF, num_folds))
y_train_folds = np.array(np.array_split(y, num_folds))

for c, gamma in itertools.product(c_intervals, gamma_intervals):
	accuracies = []
	for i in range(3):
		X_train = np.concatenate(
			X_train_folds[np.arange(num_folds) != i])
		y_train = np.concatenate(
			y_train_folds[np.arange(num_folds) != i])
		X_test = X_train_folds[i]
		y_test = y_train_folds[i]
		clf = SVC(C=c, gamma=gamma, kernel = 'rbf')
		clf.fit(X_train, y_train)
		y_test_predict = clf.predict(X_test)
		num_cor = np.sum(y_test_predict == y_test)
		accuracy = float(num_cor) / len(y_test)
		accuracies.append(accuracy)
	print("-----X----")	
	print('C=%5.1f, gamma=%4.1f, accuracy = %.3f' % (c, gamma, np.mean(accuracies)))
#	for a in accuracies:
#		print(a)


for c, gamma in itertools.product(c_intervals, gamma_intervals):
	accuracies = []
	for i in range(3):
		X_train = np.concatenate(
			XF_train_folds[np.arange(num_folds) != i])
		y_train = np.concatenate(
			y_train_folds[np.arange(num_folds) != i])
		X_test = XF_train_folds[i]
		y_test = y_train_folds[i]
		clf = SVC(C=c, gamma=gamma, kernel = 'rbf')
		clf.fit(X_train, y_train)
		y_test_predict = clf.predict(X_test)
		num_cor = np.sum(y_test_predict == y_test)
		accuracy = float(num_cor) / len(y_test)
		accuracies.append(accuracy)
	print("-----XF----")	
	print('C=%5.1f, gamma=%4.1f, accuracy = %.3f' % (c, gamma, np.mean(accuracies)))
