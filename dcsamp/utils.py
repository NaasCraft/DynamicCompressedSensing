import collections
from contextlib import contextmanager
import logging
import time

import numpy


class Timer:
    def __init__(self, name: str='default'):
        self.tics = collections.defaultdict(list)
        self.times = collections.defaultdict(list)
        self.logger = logging.getLogger('Timer({})'.format(name))

    def tic(self, key: str='default'):
        t = time.time()
        self.tics[key].append(t)

        return t

    def toc(self, key: str='default'):
        t = time.time()
        assert len(self.tics[key]) > 0

        duration = t - self.tics[key][-1]
        self.times[key].append(duration)

        self.logger.debug('({}) took {:.2f}s'.format(key, duration))

        return duration

    @contextmanager
    def time(self, key: str='default', _print: bool=False):
        self.tic(key)
        yield
        self.toc(key)


MAX_PRECISION = 1e100
MIN_PRECISION = 1e-100


def trim_array(
    arr: numpy.ndarray,
    _min: float=MIN_PRECISION,
    _max: float=MAX_PRECISION
):
    arr[arr < _min] = _min
    arr[arr > _max] = _max
