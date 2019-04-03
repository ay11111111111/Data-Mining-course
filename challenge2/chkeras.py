import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

#load_dataset
tt = pd.read_csv('dm-image-challenge-train.txt', header = None)
I = tt.values.reshape(9200, 32, 32)
N = len(I)
y_train = I[:, 8:24, 8:24].reshape(N, -1)
X_train = I.copy()
X_train[:, 8:24, 8:24] = 0
X_train = X_train.reshape(N, -1)
print(X_train.shape)
print(y_train.shape)

#load test data

tt2 = pd.read_csv('dm-image-challenge-row-testdist.txt', header = None)
#I = tt.values.reshape(920, 32, 32)
N = len(I)
y_test = I[:, 8:24, 8:24].reshape(N, -1)
X_test = I.copy()
X_test[:, 8:24, 8:24] = 0
X_test = X_test.reshape(N, -1)

#
# for i in range(tt.shape[0]):
# 	plt.imshow(np.array(tt_square[i]))
# 	plt.show()

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

y_pred = model.predict(X_train[:4])

print("Keras: ")


# calculate MAE, MSE, RMSE
#print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_train, y_pred))
#print(np.sqrt(mean_squared_error(y_true, y_pred)))

for i in range(5):
	I1 = X[30].reshape(32, 32)
	I2 = I1
	I1[8:24, 8:24] = y[30].reshape(16, 16)
	I2[8:24, 8:24] = y_pred[30].reshape(16, 16)
	plt.imshow(I1.reshape(32, 32))
	plt.imshow(I2.reshape(32, 32))
	plt.show()

#tt = pd.read_csv("dm-challenge-testdist.txt", header=None)
#I_test = tt.values[:, 0:1]
#X_test = tt.values[:, 1:101]
#y_test = tt.values[:, 101:]
 #%%
#y_pred = clf.predict(X_test)
#output = np.hstack((I_test, X_test, y_pred))
#np.savetxt('linearregression.txt', output, delimiter=',', fmt='%d')
