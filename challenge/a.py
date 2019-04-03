#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%%
tt = pd.read_csv("dm-challenge-train.txt", header=None)

I_train = tt.values[:, 0]
X_train = tt.values[:, 1:101]
y_train = tt.values[:, 101:]
#%%
tt = pd.read_csv("dm-challenge-testdist.txt", header=None)
I_test = tt.values[:, 0]
X_test = tt.values[:, 1:101]
y_test = tt.values[:, 101:]
#%%
# ANIMACIA
clf = LinearRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print(mean_squared_error(y_train, y_pred))

#%%
I = 10
plt.plot(range(0, 100), X_train[I, :], color="blue")
plt.plot(range(100, 120), y_train[I, :], color="green")

plt.plot(range(100, 120), y_pred[I, :], color="red")

plt.show()

#%%
y_pred = clf.predict(X_test)
output = np.hstack(I)
np.savetxt('dm-zeros-int.txt', y_pred, delimiter=',', fmt='%d')
