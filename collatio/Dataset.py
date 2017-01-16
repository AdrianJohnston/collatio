import numpy as np


class Dataset(object):

    def __init__(self, X, Y, shuffle=True, seed=1337):
        self.X = X
        self.Y = Y
        self.indices = None

    def __load_dataset(self, path):
        raise NotImplementedError()

    @staticmethod
    def compute_indices(num_samples, shuffle=True, seed=0):
        indices = np.arange(0, num_samples)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        return indices





