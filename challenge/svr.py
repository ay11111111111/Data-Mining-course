#%%
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%%
tt = pd.read_csv("dm-challenge-train.txt", header=None)

I = tt.values[:, 0:1]
X = tt.values[:, 1:101]
y = tt.values[:, 101:]
print(I.shape, X.shape, y.shape)


I_folds = np.split(I, 5)
X_folds = np.vsplit(X, 5)
y_folds = np.vsplit(y, 5)

I_test = I_folds[4]
X_test = X_folds[4]
y_test = y_folds[4]
print(I_test.shape, X_test.shape, y_test.shape)


I_train = I_folds[0]
X_train = X_folds[0]
y_train = y_folds[0]
for i in range(3):
    I_train = np.concatenate((I_train, I_folds[i+1]), axis = 0)
    X_train = np.vstack((X_train, X_folds[i+1]))
    y_train = np.vstack((y_train, y_folds[i+1]))
print(I_train.shape, X_train.shape, y_train.shape)

clf = SVR()
clf = MultiOutputRegressor(clf)
clf.fit(X_train, y_train, sample_weight=None)
y_pred = clf.predict(X_test)
#y2_pred = clf.predict(X_train)

print(mean_squared_error(y_test, y_pred))



##

tt = pd.read_csv("dm-challenge-testdist.txt", header=None)
I_test = tt.values[:, 0:1]
X_test = tt.values[:, 1:101]
y_test = tt.values[:, 101:]
 #%%
y_pred = clf.predict(X_test)



output = np.hstack((I_test, X_test, y_pred))
np.savetxt('knn-4-distance.txt', output, delimiter=',', fmt='%d')

#print(mean_squared_error(y_train, y2_pred))
