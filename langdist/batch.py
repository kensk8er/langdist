# -*- coding: UTF-8 -*-
"""
Define classes related to batch processing here.
"""
from copy import deepcopy
from random import shuffle

__author__ = 'kensk8er1017@gmail.com'


class BatchGenerator(object):
    """
    BatchGenerator class cretates a batch iterator on which you can iterate in order to get batches.

    Basic Usage:
        batch_generator = BatchGenerator(X, y, batch_size=128)

        for X_batch, y_batch in batch_generator:
            # it keeps iterating on the batches unless you run it with `validation=True`
            do_something_on_batch(X_batch, y_batch)
    """

    def __init__(self, X, batch_size, shuffle=True):
        """
        Constructor

        :param X: list of samples (list of lists of word_ids)
        :param batch_size: the size of samples in a batch
        :param shuffle: if True, shuffle the data in every new epoch
        """
        assert isinstance(X, list), 'Invalid argument type type(X) = {}'.format(type(X))
        assert batch_size > 0, 'batch_size <= 0'

        self._X = deepcopy(X)  # BatchGenerator shouldn't have a by-product
        self._batch_id = 0
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._data_size = len(self._X)

    def __iter__(self):
        return self

    def __next__(self):
        """
        This is called everytime you iterate on this object.

        :return: a batch of X
        """
        X = self._gen_batch(self._X, self._batch_id, self._batch_size, self._data_size)
        self._batch_id += 1
        return X

    def _gen_batch(self, X, batch_id, batch_size, data_size):
        """Generate batch for given X, batch_id, batch_size, and data_size."""
        start_index = (batch_id * batch_size) % data_size
        end_index = ((batch_id + 1) * batch_size) % data_size

        if start_index < end_index:
            return deepcopy(X[start_index: end_index])
        else:  # executing here means you have gone over X and y already
            X_first = deepcopy(X[start_index:])

            # shuffle X and y after going over them if shuffle is True
            if self._shuffle:
                shuffle(X)

            X_second = deepcopy(X[:end_index])
            return X_first + X_second
