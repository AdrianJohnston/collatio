import numpy as np

from Dataset import Dataset
import skimage.io as skio


class ImageDataset(Dataset):


    def __init__(self, X, Y, shuffle=True, seed=1337):

        Dataset.__init__(self, X, Y, shuffle, seed)

        self.__flatten_records()
        self.indices = Dataset.compute_indices(len(self.X_flat), shuffle=shuffle, seed=seed)


    def __flatten_records(self):

        X_flat = []
        y_flat = []
        lens = []
        for i in range(len(self.X)):
            X_flat += [x for x in self.X[i]]
            y_flat += [self.Y[i] for y in range(len(self.X[i]))]

        self.X_flat = X_flat
        self.Y_flat = y_flat
        assert len(X_flat) == len(y_flat)


    def read_data(self, index):
        return np.array(skio.imread(self.X_flat[self.indices[index]])), np.array(self.Y_flat[self.indices[index]])


    def read_data_absolute(self, index):
        '''
        Reads the absolute data value. By passing indexed views. E.g. reads from the sorted data array
        :param index:
        :return: np.array for the image
        '''
        print("Loading:", self.X_flat[index])
        return np.array(skio.imread(self.X_flat[index])), np.array(self.Y_flat[index])

    def get_indexed_path(self, index):
        return self.X_flat[self.indices[index]]

    def read_data_slice(self, start, stop, step):

        x_slice = []
        y_slice = []
        if step == None:
            step = 1

        for i in range(start, stop, step):
            x, y = self.read_data(i)
            x_slice += [x]
            y_slice += [y]
        return np.array(x_slice), np.array(y_slice)

    def read_data_absolute_indices(self, indices):
        '''
        Reads the data from the original data store, bypassing self.indices
        :param indices:
        :return:
        '''
        if isinstance(indices, list):

            x_slice = []
            y_slice = []
            for i in indices:
                x, y =self.read_data_absolute(i)
                x_slice += [x]
                y_slice += [y]

            return np.array(x_slice), np.array(y_slice)
        else:
            raise TypeError("Error, read_absolute_data takes a list as input")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        if isinstance(item, slice):
            return self.read_data_slice(item.start, item.stop, item.step)

        else:
            return self.read_data(item)