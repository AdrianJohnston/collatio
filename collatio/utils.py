from __future__ import print_function

import numpy as np
#from Phorcys import *
import os, sys
import scandir
import cPickle
import simplejson as json
import gzip

def file2labels(filename):

    filename = os.path.split(filename)[-1]
    c, id, view = filename.split('.')[0].split('_')
    return c, id, view

def write_json_dataset(filename, dataset, compress=False):

    if not '.json' in filename:
        filename += '.json'


    if compress and not '.gz' in filename:
        filename += '.gz'

    if compress:
        if not '.gz' in filename:
            filename += '.gz'

        with gzip.open(filename, 'wb') as f:
            json.dump(dataset, f, indent=4 * ' ')

    else:

        f = file(filename, 'wb')
        json.dump(dataset, f, indent=4*' ')
        f.close()


def load_json_dataset(filename):

    if not '.json' in filename:
        filename += '.json'

    if 'gz' in filename:
        #Load with gzip
        with gzip.open(filename, 'rb') as f:
            dataset = json.load(f)
    else:
        f = file(filename, 'rb')
        dataset = json.load(f)
        f.close()

    return dataset


def save_pkl_dataset(filename, dataset):

    if not '.pkl' in filename:
        filename += '.pkl'

    f = file(filename, 'wb')
    cPickle.dump(dataset,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load_pkl_dataset(filename):

    if not '.pkl' in filename:
        filename += '.pkl'

    f = file(filename, 'rb')
    dataset = cPickle.load(f)
    f.close()
    return dataset


def cache_png_dataset(path):
    dir_iter = scandir.scandir(path)

    file_count = 0

    Y = []
    X = []

    data_paths = []

    for it in dir_iter:
        file_count += 1

        if os.path.isdir(it.path):
            data_paths.append(it.path)

    for d in data_paths:

        dir_cache = d + ".cache"
        print (dir_cache)

        cache = open(dir_cache, 'w')
        # m_list = []
        # i_list = []
        for it in scandir.scandir(d):
            cache.write(it.path + "\r\n")
            print(it.path)

        cache.flush()
        cache.close()

    return X, Y


def check_cache_exist(path):

    if os.path.isdir(path):
        if os.path.exists(path+"cache"):
            return True
        else:
            return False
    else:
        raise IOError("Input to check_cache_exist must be a directory")

def load_cache(path):

    paths = []
    f =  open(path, 'r')
    lines = f.readlines()
    for l in lines:
        paths.append(l.strip())
    f.close()
    return paths


def cache2dataset(cache):

    X = []
    Y = []
    ids = []
    views = []
    cache.sort()
    data = {}

    #Allocate the buffer
    for p in cache:
        y, id, v = file2labels(p)
        data[id] = []

    #Fill the buffer
    for p in cache:
        y, id, v = file2labels(p)
        data[id].append([p, y, id, v])

    return data


def organise_cache_data(cache_data):
    X = []
    Y = []
    ids = []
    views = []

    for k, v in cache_data.iteritems():

        X_sub = []
        Y_sub = []
        id_sub = []
        view_sub = []
        for x in v:
            X_sub.append(x[0])
            Y_sub.append(x[1])
            id_sub.append(x[2])
            view_sub.append(x[3])

        X.append(X_sub)
        Y.append(Y_sub)
        ids.append(id_sub)
        views.append(view_sub)

    Y = np.array(Y)[:, 0]
    ids = np.array(ids)
    views = np.array(views)
    return X, Y, ids, views