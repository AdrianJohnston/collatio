from __future__ import print_function

from collatio.ArrayDataset import *


X=np.random.random((10,3))
Y=np.arange(0,10).reshape((10, 1))

dataset = ArrayDataset(X=X, Y=Y)

assert dataset.X.shape == (10, 3)
assert dataset.Y.shape == (10, 1)


dataset = ArrayWrapper(X=X)
assert dataset.X.shape == (10, 3)