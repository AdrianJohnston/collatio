import numpy as np


class BatchIterator(object):

    def __init__(self, dataset, batch_size, callback, start=0, stop=-1, step=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.start = start
        self.stop = (self.num_samples if stop == -1 else stop)
        self.step = (step if step >= 1 else 1)
        self.current_batch = 0

    def __call__(self, *args, **kwargs):
        return self.__iter__()

    @property
    def num_batches(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    @staticmethod
    def slice_size(slice):
        return (slice.stop - slice.start) / slice.step

    def __iter__(self):

        for i in range(self.num_batches):
            yield next(self)

    def next(self):

        num_samples = self.num_samples
        batch_size = self.batch_size
        i = self.current_batch
        _slice = slice(i * batch_size, np.minimum((i + 1) * batch_size, num_samples))
        return self.dataset[_slice] + (i,)


