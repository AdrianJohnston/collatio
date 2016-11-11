import numpy as np


def train_test_iterators(dataset, batch_size, split=0.8):

    num_samples = len(dataset)
    stop = int(np.floor(split * num_samples))
    train_it = TrainTestIterator(dataset, batch_size, start=0, stop=stop)
    test_it = TrainTestIterator(dataset, batch_size, start=stop, stop=num_samples)
    return train_it, test_it


class TrainTestIterator(object):

    def __init__(self, dataset, batch_size, start=0, stop=-1, step=1):
        self.dataset = dataset
        self.batch_size = batch_size
        num_samples = len(dataset)
        self.start = start
        self.stop = (num_samples if stop == -1 else stop)
        self.step = (step if step >= 1 else 1)
        self.num_samples = stop - start

    def __call__(self, *args, **kwargs):
        return self.batch_iterator()

    @staticmethod
    def calculate_num_batches(num_examples, batchsize):
        return int(np.ceil(num_examples / float(batchsize)))

    @property
    def num_batches(self):
        print("num samples", self.num_samples)
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    @staticmethod
    def slice_size(slice):
        return (slice.stop - slice.start) / slice.step

    def batch_iterator(self):

        num_samples = self.num_samples
        batch_size = self.batch_size
        num_batches = self.num_batches

        #TODO: Select batches using slice e.g. only get the first 10 batches iterator()[0:10]

        for i in range(0, num_batches):
            _slice = slice(self.start + (i * batch_size), np.minimum((i + 1) * batch_size, num_samples))
            #yield X[_slice], (y[_slice] if y is not None else None)
            yield self.dataset[_slice] + (i,)