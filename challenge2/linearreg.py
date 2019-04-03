import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

#load_dataset
tt = pd.read_csv('dm-image-challenge-train.txt', header = None)

I = tt.values.reshape(9200, 32, 32)
N = len(I)

y = I[:, 8:24, 8:24].reshape(N, -1)
X = I.copy()

X[:, 8:24, 8:24] = 0
X = X.reshape(N, -1)

print(X.shape)
print(y.shape)


#
# for i in range(tt.shape[0]):
# 	plt.imshow(np.array(tt_square[i]))
# 	plt.show()

clf = LinearRegression()
clf.fit(X, y)
y_pred = clf.predict(X)

print("Linear Regression: ")


# calculate MAE, MSE, RMSE
#print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y, y_pred))
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
