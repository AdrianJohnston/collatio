# collatio
Dataset utilities for deep learning

Features include:

* Batched Iteration
    * provide a callback to do preprocessing
* Multiprocessed Batch Iteration
    * provide a callback to do preprocessing in multiple processes
* Dataset classes for easy interop with iterators
    * Array Wrapper - single Array
    * Array Dataset - X, Y pairs (e.g. features, labels)
    * Folder Dataset - Quick indexing (using scandir) of folders for loading large collections of files. Useful for image sets
        * Includes lazy loading of paths, provide a load_fn to do custom data loading e.g. lambda x: skimage.io.imread(x)