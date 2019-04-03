import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

A = pd.read_csv('lda-clf.txt', header=None)
X = A.values[:, 0:2]
y = A.values[:, 2]

I = y == 1
J = [not x for x in I]

#LDA
m1 = np.mean(X[I,:],axis=0)
m2 = np.mean(X[J,:],axis=0)

e1 = 1 / (len(X[I,:])-1)
e2 = 1 / (len(X[J,:])-1)

s1 = np.dot((X[I,:]-m1).transpose(), X[I,:]) / e1
s2 = np.dot((X[J,:]-m1).transpose(), X[J,:]) / e2

sw = s1 + s2
w = np.dot(np.linalg.inv(sw), (m2-m1))

print("The coeffs are:", w)

clf = LDA()
clf.fit(X, y)

print(np.vstack((clf.predict(X), y)).T)
plt.plot(X[I,0], X[I,1], '.')
plt.plot(X[J,0], X[J,1], '.')
plt.grid()
plt.show()


