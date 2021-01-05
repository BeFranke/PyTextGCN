from collections import Iterable
from typing import List

import numpy as np


class TextGCNBatcher(Iterable):
    def __init__(self, X: List[str], y: List[int], shuffle: bool = True, batch_size: int = 1024):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.y = np.array(y, dtype=object)
        self.X = np.array(X, dtype=object)
        self.ind = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.ind)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        max_i = np.minimum(len(self.X), self.batch_size * self.i)
        inds = self.ind[:max_i]
        return self.X[inds].tolist(), self.y[inds].tolist()

    def __len__(self):
        return int(len(self.X) / self.batch_size)
