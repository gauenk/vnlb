"""
Timer module

code copied from Ross Girshick on Faster-RCNN
"""

import time
import functools
from contextlib import ContextDecorator

VERBOSE_TIMING = False

class Timer():
    """A simple timer."""
    def __init__(self,name="timer"):
        self.name = name
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def __call__(self, func):
        """Support using Timer as a decorator"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper_timer

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.tic()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.toc()
        if VERBOSE_TIMING:
            print("[%s]: dtime = %2.2e"%(self.name,self.diff))

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
    def __str__(self):
        return "-=-=- Timer Info -=-=-\n\
        Total Time: {}\n\
        Calls: {}\n\
        Start Time: {}\n\
        Diff: {}\n\
        Average Time: {}\n".format(self.total_time,
                                   self.calls,
                                   self.start_time,
                                   self.diff,
                                   self.average_time)
