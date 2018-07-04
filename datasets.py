from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.model_selection import train_test_split


seed = 1337


def one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class MSCOCODataSet:

    def __init__(self):
        pass


class DataIterator:

    def __init__(self, x, y, batch_size, label_off=False):
        self.x = x
        self.label_off = label_off
        if not self.label_off:
            self.y = y
        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert (self.batch_size <= self.num_examples)

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.x = self.x[perm]
            if not self.label_off:
                self.y = self.y[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer

        if not self.label_off:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()
