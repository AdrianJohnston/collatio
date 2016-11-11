from __future__ import print_function
import numpy as np

from Dataset import Dataset


class ArrayWrapper(Dataset):
    '''
    Helper class for wrapping an array for use with MultiProcessIterator
    '''
    def __init__(self, X, shuffle=True, seed=1337):
        super(ArrayWrapper, self).__init__(X=X, Y=None, shuffle=shuffle, seed=seed)
        self.num_samples = X.shape[0]
        self.indices = ArrayWrapper.compute_indices(self.num_samples)


    @staticmethod
    def compute_indices(num_samples, shuffle=True, seed=0):
        indices = np.arange(0, num_samples)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        return indices

    def __getitem__(self, item):
        return self.X[item]

    def __len__(self):
        return self.num_samples


class ArrayDataset(Dataset):
    '''
    Helper class for wrapping an array dataset (e.g. X, Y pairs) for use with MultiProcessIterator

    '''
    def __init__(self, X, Y, shuffle=True, seed=1337):
        super(ArrayDataset, self).__init__(X=X, Y=Y, shuffle=shuffle, seed=seed)
        self.num_samples = X.shape[0]
        self.indices = ArrayDataset.compute_indices(self.num_samples)


    @staticmethod
    def compute_indices(num_samples, shuffle=True, seed=0):
        indices = np.arange(0, num_samples)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        return indices


    def __getitem__(self, item):
        return self.X[self.indices[item]], self.Y[self.indices[item]]

    def __len__(self):
        return self.num_samples