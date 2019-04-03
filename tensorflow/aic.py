import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from math import log

def linearReg(power):
	tt = pd.read_csv ("aic1.txt",header=None)
	x=tt.values [:,0]
	y=tt.values [:,1]
	x = x[:,np.newaxis]
	X = np.hstack ((0*x + 1, x))
	if power >= 2:
		for i in range(2, power):
			X = np.hstack((X, x ** i))
		print(X)
	clr = linear_model.LinearRegression()

	clr.fit(X,y)
	y_pred = clr.predict(X)
	return y, y_pred

#calculate AIC
def calcAIC():
	y, y_pred = linearReg(1)
	N = len(y)
	aic = 100000000000
	index = 0
	for p in range(1, 12):
		y, y_pred = linearReg(p)
		N = len(y)

		E = np.sum((y - y_pred)**2)
		a = N * log(E / N) + 2 * (p + 2)
		print(p)
		print(a)
		if a < aic:
			aic = a
			index = p
	return index, aic

print( "p=%f: %f" % calcAIC())
