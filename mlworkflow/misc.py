import os
import re
import pickle
import warnings
from datetime import datetime
from collections import deque
from functools import wraps, partial
from multiprocessing.pool import ThreadPool


_NOVALUE = object()


def gen_id(filename):
    datetime_ = datetime.now()
    date = datetime_.strftime("%Y%m%d")
    time = datetime_.strftime("%H%M%S")
    datetime_ = "{}_{}".format(date, time)
    return filename.format(datetime_, datetime=datetime_, date=date, time=time)


class DictObject(dict):
    __slots__ = ("__weakref__",)

    def __setattr__(self, name, value):
        self.__setitem__(name, value)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError as e:
            raise AttributeError(*e.args) from e

    def __delattr__(self, name):
        try:
            self.__delitem__(name)
        except KeyError as e:
            raise AttributeError(*e.args) from e

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self.update(d)

    def __repr__(self):
        return "DictObject(%s)" % (super().__repr__(),)

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
    def __init__(self, pool_size=1, impl=ThreadPool):
        self.pool_size = pool_size
        self.pool = impl(pool_size)
        self.pending = deque()
        self.last_handle = None

    def __len__(self):
        return self.pool_size

    def ready(self):
        return self.last_handle is None or self.last_handle.ready()

    def do(self, f, *args, **kwargs):
        self.last_handle = handle = self.pool.apply_async(partial(f, *args, **kwargs))
        return handle

    def wait_for_complete(self, i):
        j = i+1 if i != -1 else None
        for p in self.pending[i:j]:
            p.wait()

    def run_async(self, f, *args, **kwargs):
        handle = self.pool.apply_async(partial(f, *args, **kwargs))
        self.pending.append(handle)
        return handle

    def collect_runs(self):
        lst = [handle.get() for handle in self.pending]
        self.pending.clear()
        return lst

    def yield_async(self, gen, in_advance=1):
        assert isinstance(self.pool, ThreadPool), "yield_async is only allowed with ThreadPool implementation"
        if self.pool_size > 1:
            warnings.warn("Avoid using more than 1 thread with a generator")
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
        self.pool.terminate()
        self.pool.join()
