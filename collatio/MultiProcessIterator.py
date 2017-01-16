import multiprocessing as mp

import numpy as np

from BatchIterator import BatchIterator


class MultiProcessIterator(object):

    def __init__(self, dataset, batch_size, callback=lambda data:data, start=0, stop=-1, step=1,
                 num_workers=1, wait_time=0.001, cache_size=10):

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.start = start
        self.stop = (self.num_samples if stop == -1 else stop)
        self.step = (step if step >= 1 else 1)
        self.callback = callback
        self.num_workers = num_workers
        self.workers = mp.Pool(num_workers)
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
        num_workers = self.num_workers
        result_slices = []
        start = i * batch_size
        for j in range(0, num_workers):

            # (i *batch_size): batch_size//num_workers
            mp_batch_size = batch_size // num_workers
            start_offset = j * mp_batch_size

            if (j+1) >= num_workers:
                end_offset = start + start_offset + batch_size - (j * mp_batch_size)
            else:
                end_offset = start + start_offset + mp_batch_size

            begining = start + start_offset
            #make sure to take what ever is left, if it is not cleanly divisible
            _slice = slice(begining, np.minimum(end_offset, num_samples))
            result_slices.append(_slice)

        data_chunks = []
        res_chunks = []

        for s in result_slices:
            data_chunks.append(self.dataset[s])


        multiple_results = []
        for n in range(num_workers):
            if len(data_chunks[n][0]) > 0:
                multiple_results += [self.workers.apply_async(self.callback, (data_chunks[n],))]


        # multiple_results = [self.workers.apply_async(self.callback, (data_chunks[n],)) for n in range(num_workers)]
        results = [res.get() for res in multiple_results]

        final_results = []

        if len(results) == 0:
            return ([], [], i)

        for j in range(len(results[0])):
            r = []
            for i in range(len(results)):
                r.append(results[i][j])

            final_results.append(np.concatenate(r, axis=0))
            # print(final_results[j].shape)

        final_results = tuple(final_results)
        i = self.current_batch
        self.current_batch += 1
        data = final_results
        return data + (i,)

    def close(self):
        self.workers.close()

    def __len__(self):
        return self.num_batches



      