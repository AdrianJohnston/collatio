
import numpy as np
import scandir
from Dataset import Dataset

import skimage.io as skio

class InstanceFolderDataset(Dataset):
    '''
    Type of labeled folder dataset
    Rather than the traditional structure:
        - train
            -- class_id
                -- Item
    It follows the following structure:
        - train
            -- class id
                -- instance id 
                    -- Item
    '''

    #TODO: Add glob for file type
    def __init__(self, path, load_fn=None, index_fn=None, shuffle=True, seed=1337, flatten=False, filetypes=[".png", ".jpg", ".jpeg"]):

        self.filetypes = filetypes
        if index_fn is None:
            X, Y , iid = self.index_dir(path, filetypes)
        else:
            X, Y, iid = index_fn(path, filetypes)
        
        #X, Y, iid = self.sort_by_label(X, Y, iid)
        self.iid = iid
        super(InstanceFolderDataset, self).__init__(X=X, Y=Y, shuffle=shuffle, seed=seed)
        self.flatten = flatten
        if flatten:
            self.__flatten_records()

        self.indices = Dataset.compute_indices(len(self.X), shuffle=shuffle, seed=seed)
        if load_fn is None:
            load_fn = lambda x : x
        
        self.load_fn = load_fn

    def sort_by_label(self, X, Y, iid):

        idx = np.argsort(Y)
        X = np.array(X)[idx]
        Y = np.array(Y)[idx]
        iid = np.array(iid)[idx]


        return list(X), list(Y), list(iid)


    def verify_labels(self):
        for x, y in zip(self.X ,self.Y):
            if not InstanceFolderDataset.split_label(x) == y:
                print(x, y)

    @staticmethod
    def index_dir(path, filetypes):
        dir_iter = scandir.scandir(path)

        file_count = 0

        Y = []
        X = []

        class_paths = []

        for it in dir_iter:
            file_count += 1
            class_paths.append(it.path)

        num_models = 0

        instance_paths = []
        for c  in class_paths:
            for it in scandir.scandir(c):
                instance_paths.append(it.path)

        iids = []
        for i in instance_paths:
        
            m_list = []
            iids_list = []
            ys_list = []

            for it in scandir.scandir(i):
                num_models += 1
                m_list.append(it.path)
            
            m_list = [file for file in m_list for ft in filetypes if ft in file]
            
            p_split = i.split('/')
            iids_list.append(p_split[-1])
            ys_list.append(int(p_split[-2]))
            
            X.append(m_list)
            iids.append(iids_list)
            Y.append(ys_list)
            
        return X, Y, iids

    def __flatten_records(self):

        X_flat = []
        y_flat = []
        iid_flat = []
        lens = []
        for i in range(len(self.X)):
            X_flat += [x for x in self.X[i]]
            y_flat += [y for y in self.Y[i]]
            iid_flat += [self.iid[i] for i in self.X[i]]


        self.X_structured = self.X
        self.Y_structured = self.Y
        self.iid_structured = self.iid

        self.X = X_flat
        self.Y = y_flat
        self.iid = iid_flat
        assert len(self.X) == len(self.Y)
        assert len(self.X) == len(self.iid)

    @staticmethod
    def split_label(x):
        return int(x.split('/')[-2])

    def read_data(self, index):
        '''
        Calls self.load_fn on each path in self.X. Should return numpy array or other indexable data structure
        :param index: The index of the sample to load. Note: This loads X[indices[index]]
        :return: load_fn(X), Y
        '''
        return self.load_fn((np.array(self.X[self.indices[index]]), np.array(self.Y[self.indices[index]]), np.array(self.iid[self.indices[index]])))

    def read_data_absolute(self, index):
        '''
        Reads the absolute data value. By passing indexed views. E.g. reads from the sorted data array
        :param index:
        :return: np.array for the image
        '''
        return np.array(skio.imread(self.X[index])), np.array(self.Y[index]), np.array(self.iid[index])

    def get_indexed_path(self, index):
        return self.X[self.indices[index]]


    def read_data_slice(self, start, stop, step):

        x_slice = []
        y_slice = []
        i_slice = []
        if step == None:
            step = 1

        for i in range(start, stop, step):
            x, y, iid = self.read_data(i)
            x_slice += [x]
            y_slice += [y]
            i_slice += [iid]
        return np.array(x_slice), np.array(y_slice), np.array(i_slice)


    def read_data_absolute_indices(self, indices):
        '''
        Reads the data from the original data store, bypassing self.indices
        :param indices:
        :return:
        '''
        if isinstance(indices, list):

            x_slice = []
            y_slice = []
            i_slice = []

            for i in indices:
                x, y, iid = self.read_data_absolute(i)
                x_slice += [x]
                y_slice += [y]
                i_slice += [iid]

            return np.array(x_slice), np.array(y_slice), np.array(i_slice)
        else:
            raise TypeError("Error, read_absolute_data takes a list as input")



    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        if isinstance(item, slice):
            return self.read_data_slice(item.start, item.stop, item.step)

        else:
            return self.read_data(item)
