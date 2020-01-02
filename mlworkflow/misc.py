from multiprocessing.pool import ThreadPool
from datetime import datetime
from collections import deque
from functools import wraps
import pickle
import os
import re


_NOVALUE = object()


def gen_id(filename):
    datetime_ = datetime.now()
    date = datetime_.strftime("%Y%m%d")
    time = datetime_.strftime("%H%M%S")
    datetime_ = "{}_{}".format(date, time)
    return filename.format(datetime_, datetime=datetime_, date=date, time=time)


class DictObject(dict):
    def __new__(cls, *args, **kwargs):
        dict_object = super().__new__(cls, *args, **kwargs)
        dict_object.__dict__ = dict_object
        return dict_object

    def __repr__(self):
        return "{}({})".format(self.__class__.__qualname__,
                               super().__repr__())

    @classmethod
    def from_dict(cls, dic):
        """__init__ may not simply copy the argument in the object. In order
        to directly feed the dictionary, from_dict can be used
        """
        dict_object = cls.__new__(cls)
        dict_object.update(dic)
        return dict_object


def naturally_sorted(lst):
    def _key(item):
        key = re.split(r"([0-9]+(?:\.[0-9]+)?)", item)
        for i in range(1, len(key), 2):
            key[i] = float(key[i])
        return key
    return sorted(lst, key=_key)


def pickle_cache(path):
    def _decorator(f):
        @wraps(f)
        def wrapper(**kwargs):
            filename = path.format(**kwargs)
            if os.path.exists(filename):
                with open(filename, "rb") as file:
                    return pickle.load(file)
            result = f(**kwargs)
            with open(filename, "wb") as file:
                pickle.dump(result, file)
            return result
        return wrapper
    return _decorator


class SideRunner:
    def __init__(self):
        self.pool = ThreadPool(1)
        self.pending = deque()

    def run_async(self, f):
        handle = self.pool.apply_async(f)
        self.pending.append(handle)
        return handle

    def wait_for_complete(self, i):
        j = i+1 if i != -1 else None
        for p in self.pending[i:j]:
            p.wait()

    def collect_runs(self):
        lst = [handle.get() for handle in self.pending]
        self.pending.clear()
        return lst

    def yield_async(self, gen, in_advance=1):
        pending = deque()
        def consume(gen):
            return next(gen, _NOVALUE)
        for _ in range(in_advance):
            pending.append(self.pool.apply_async(consume, args=(gen,)))
        while True:
            pending.append(self.pool.apply_async(consume, args=(gen,)))
            p = pending.popleft().get()
            if p is _NOVALUE:
                break
            yield p

    def __del__(self):
        self.pool.close()
        self.pool.join()
