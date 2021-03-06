
import numpy as np
import scandir
from Dataset import Dataset

import skimage.io as skio

class FolderDataset(Dataset):

    '''
    Flat Folder Dataset loader
    '''
    default_glob = ['*.png', '*.jpg']
    def __init__(self, path, load_fn, split_label_fn=None, shuffle=True, seed=1337):

        if split_label_fn is None:
            split_label_fn = self.default_split_label

        self.split_label_fn = split_label_fn
        X, Y = self.index_dir(path, split_label_fn)
        super(FolderDataset, self).__init__(X=X, Y=Y, shuffle=shuffle, seed=seed)
        self.indices = Dataset.compute_indices(len(self.X), shuffle=shuffle, seed=seed)
        self.load_fn = load_fn

    def default_split_label(self, file):
        return file.split('/')[-1].split('.')[0]

    def index_dir(self, path, split_label_fn):
        dir_iter = scandir.scandir(path)

        file_count = 0

        Y = []
        X = []

        files = []
        labels = []
        for it in dir_iter:
            file_count += 1
            files.append(it.path)
            l = split_label_fn(it.path)
            labels.append(l)

        return files, labels

    def __flatten_records(self):

        X_flat = []
        y_flat = []
        lens = []
        for i in range(len(self.X)):
            X_flat += [x for x in self.X[i]]
            y_flat += [self.Y[i] for y in range(len(self.X[i]))]

        self.X_structured = self.X
        self.Y_structured = self.Y
        self.X = X_flat
        self.Y = y_flat
        assert len(self.X) == len(self.Y)

    @staticmethod
    def split_label(x):
        return int(x.split('/')[-2])

    def read_data(self, index):
        '''
        Calls self.load_fn on each path in self.X. Should return numpy array or other indexable data structure
        :param index: The index of the sample to load. Note: This loads X[indices[index]]
        :return: load_fn(X), Y
        '''
        return np.array(self.load_fn(self.X[self.indices[index]])), np.array(self.Y[self.indices[index]])

    def read_data_absolute(self, index):
        '''
        Reads the absolute data value. By passing indexed views. E.g. reads from the sorted data array
        :param index:
        :return: np.array for the image
        '''
        return np.array(self.load_fn(self.X[index])), np.array(self.Y[index])

    def get_indexed_path(self, index):
        return self.X[self.indices[index]]


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
                x, y = self.read_data_absolute(i)
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
        pass