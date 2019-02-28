import os
import sys
import logging
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'state_next', 'done'])


class Memory(object):
    def __init__(self, capacity=100000, replace=False, **kwargs):
        self._buffer = deque(maxlen=capacity)
        self._replace = replace
        self._fields = Transition._fields
        logging.info("Memory, capacity[%d], replace[%d]", capacity,
                     self._replace)

    def add(self, record):
        self._buffer.append(record)

    def sample_indices(self, batch_size):
        indices = np.random.choice(
            self.size, size=batch_size, replace=self._replace)
        return indices

    def sample_batch(self, batch_size):
        assert self.size >= batch_size
        indices = self.sample_indices(batch_size)
        batch = {
            f: np.array([getattr(self._buffer[i], f) for i in indices])
            for f in self._fields
        }
        return batch

    @property
    def size(self):
        return len(self._buffer)

    def __len__(self):
        return self.size
