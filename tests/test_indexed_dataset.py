from __future__ import print_function
from collatio.Dataset import Dataset
from collatio.BatchIterator import BatchIterator
import numpy as np

class IndexedDataset(Dataset):

    def __init__(self, index_file, load_fn = None, shuffle=True, seed=1337):
        self.index_file = index_file
        self.X = self.__read_index_file()
        self.num_samples = self.X.shape[0]
        super(IndexedDataset, self).__init__(X=self.X, Y=None, shuffle=shuffle, seed=seed)
        self.indices = Dataset.compute_indices(len(self.X), shuffle=shuffle, seed=seed)

        if load_fn is None:
            load_fn = lambda x: x

        self.load_fn = load_fn

    def __read_index_file(self):

        rows = []
        with open(self.index_file, 'rb') as f:
            for l in f.readlines():
                l = l.strip('\n')
                rows.append(tuple(l.split(' ')))

        return np.array(rows)

    def read_data(self, index):
        '''
        Calls self.load_fn on each path in self.X. Should return numpy array or other indexable data structure
        :param index: The index of the sample to load. Note: This loads X[indices[index]]
        :return: load_fn(X), Y
        '''
        return self.load_fn(self.X[self.indices[index]])

    def read_data_slice(self, start, stop, step):

        x_slice = []
        if step == None:
            step = 1

        for i in range(start, stop, step):
            x = self.read_data(i)
            x_slice += [x]

        return tuple(x_slice[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        if isinstance(item, slice):
            return self.read_data_slice(item.start, item.stop, item.step)

        else:
            return self.read_data(item)


def callback(x):
    print("asdasd", x)
    return x

def load_fn(x):
    x1, x2, x3 = x
    return x1 ,x2, int(x3)

if __name__ == "__main__":

    test_file = "index_test.txt"
    id = IndexedDataset(index_file=test_file, load_fn=load_fn)

    iterator = BatchIterator(id, batch_size=1, callback=callback)
    print(id[0])
    for x in iterator:
        print(x)

