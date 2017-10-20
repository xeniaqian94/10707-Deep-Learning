import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
random.seed(int(time.time()))

data=np.genfromtxt(sys.argv[1],delimiter=",")
print data.shape
X=data[:,:-1]
Y=data[:,-1]
print X.shape
print Y.shape
print type(Y)

x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
grid = X[int(random.random()*len(X))].reshape(np.sqrt(X.shape[1]),np.sqrt(X.shape[1]))
print grid

print grid.T

plt.imshow(grid,cmap='gray')
plt.show()
plt.cla()


plt.imshow(grid.T,cmap='gray')
plt.show()
plt.cla()


