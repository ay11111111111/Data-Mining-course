import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster
from scipy.signal import convolve2d

image = plt.imread('barbara.png')

h = np.array([[-1, -1], [1, 1] ])
#h = np.array([[1, 1, 1, 1, 1], [ 1, 1, 1, 1, 1]])
#h = np.ones ((10,10))

y = convolve2d(image, h, mode = 'same')
x = convolve2d(image, h.T , mode = 'same')
plt.figure(1)
plt.imshow(image, interpolation = 'nearest', cmap = 'gray')


plt.figure(2)
plt.imshow(y, interpolation = 'nearest', cmap = 'gray')

plt.figure(3)
plt.imshow(x, interpolation = 'nearest', cmap = 'gray')

plt.show()
