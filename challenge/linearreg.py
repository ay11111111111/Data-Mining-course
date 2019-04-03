import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error


tt = pd.read_csv('dm-challenge-train.txt', header = None)
X_train = tt.values[:, 1:101]
y_train = tt.values[:, 101:121]

clf = LinearRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)

print("Linear Regression: ")
# calculate MAE, MSE, RMSE
#print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_train, y_pred))
#print(np.sqrt(mean_squared_error(y_true, y_pred)))



tt = pd.read_csv("dm-challenge-testdist.txt", header=None)
I_test = tt.values[:, 0:1]
X_test = tt.values[:, 1:101]
y_test = tt.values[:, 101:]
 #%%
y_pred = clf.predict(X_test)
output = np.hstack((I_test, X_test, y_pred))
np.savetxt('linearregression.txt', output, delimiter=',', fmt='%d')
