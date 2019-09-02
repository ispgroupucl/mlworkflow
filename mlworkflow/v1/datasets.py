from pickle import Pickler, _Unpickler as Unpickler
from abc import ABCMeta, abstractmethod
from collections import ChainMap, namedtuple
import numpy as np
import functools
import sys
import os

import weakref


def chunkify(iterable, n, skip_incomplete=False):
    """Return a generator providing chunks (lists of size n) of the iterable.

    >>> tuple(chunkify(range(10), 5))  # len(iterable) % n == 0
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    >>> tuple(chunkify(range(12), 5))  # len(iterable) % n != 0
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11])
    >>> tuple(chunkify([], 100))       # Empty iterable example
    ([],)
    """
    offset = 0
    ret = [None]*n  # filled the majority of the time => avoid growing list
    i = -1
    for i, e in enumerate(iterable):
        if i - offset == n:  # yield complete sublist and create a new list
            yield ret
            offset += n
            ret = [None]*n
        ret[i - offset] = e
    last_n = i-offset+1
    if last_n == n or not skip_incomplete:
        yield ret[:last_n]  # yield the incomplete subset ([] if i = -1)


class Dataset(metaclass=ABCMeta):
    """The base class for any dataset, provides the and batches methods from
    gen_keys() and get_item(key)

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d.query([0, 1])
    (array(['Denzel', 'Tom'], ...), array(['Washington', 'Hanks'], ...))
    >>> list(d.batches([0, 1], 1))
    [(array(['Denzel'], ...), array(['Washington'], ...),
     (array(['Tom'], ...), array(['Hanks'], ...))]

    We can see the latter provides two batches
    """

    @abstractmethod
    def gen_keys(self):
        pass

    @abstractmethod
    def get_item(self, key):
        """Returns a tuple for one item, typically (Xi, Yi), or (Xi,)
        """
        pass

    @property
    def parent_dataset(self):
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset):
        self._parent_dataset = parent_dataset
        if self._parent_dataset is not None:
            self._context = parent_dataset.context.new_child(self.context.maps[0])

    dataset = parent_dataset

    @property
    def context(self):
        try:
            return self._context
        except AttributeError:
            self._context = ChainMap()
            return self._context


class TransformedDataset(Dataset):
    """A dataset that passes yielded items through transforms

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = TransformedDataset(d, [lambda x: (x[0][:1]+".", x[1])])
    >>> d.query([0, 1])
    (array(['D.', 'T.'], ...), array(['Washington', 'Hanks'], ...))
    """
    def __init__(self, dataset, transforms=[]):
        """Creates a dataset performing operations for modifying another"""
        self.dataset = dataset
        self.transforms = [(t, getattr(t, "needs_key", False))
                           for t in transforms]

    def gen_keys(self):
        return self.dataset.gen_keys()

    def get_item(self, key):
        item = self.dataset.get_item(key)
        for transform, needs_key in self.transforms:
            if needs_key:
                item = transform(key, item)
            else:
                item = transform(item)
        return item

    def add_transform(self, transform=None, *, needs_key=False):
        _needs_key = needs_key
        def add_transform(transform):
            needs_key = _needs_key
            if not needs_key:
                needs_key = getattr(transform, "needs_key", False)
            item = (transform, needs_key)
            self.transforms.append(item)
            return transform
        if transform is not None:
            return add_transform(transform)
        return add_transform

    def add_transforms(self, transforms):
        self.transforms.extend((t, getattr(t, "needs_key", False))
                               for t in transforms)


class CacheLastDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = (None, None)

    def gen_keys(self):
        return self.dataset.gen_keys()

    def after_cache_miss(self, key, item):
        pass

    def get_item(self, key):
        cached_key, item = self.cache
        if key != cached_key:
            item = self.dataset.get_item(key)
            self.cache = (key, item)
            self.after_cache_miss(key, item)
        return item


class CacheKeysDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.keys = None

    def _gen_keys(self):
        keys = []
        for key in self.dataset.gen_keys():
            keys.append(key)
            yield key
        self.keys = tuple(keys)

    def gen_keys(self):
        if self.keys is not None:
            return self.keys
        return self._gen_keys()


class AugmentedDataset(Dataset, metaclass=ABCMeta):
    """ "Augments" a dataset in the sense that it can produce many items from
    one item of the dataset.

    >>> class PermutingDataset(AugmentedDataset):
    ...     def augment(self, root_key, root_item):
    ...         yield (root_key, 0), root_item
    ...         yield (root_key, 1), root_item[::-1]
    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = PermutingDataset(d)
    >>> new_keys = list(d.gen_keys())
    >>> new_keys
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> d.query(new_keys)
    (array(['Denzel', 'Washington', 'Tom', 'Hanks'], ...),
     array(['Washington', 'Denzel', 'Hanks', 'Tom'], ...))
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = (None, None)

    def _augment(self, root_key):
        if self.cache[0] != root_key:
            root_item = self.dataset.get_item(root_key)
            new_items = dict(self.augment(root_key, root_item))
            self.cache = (root_key, new_items)
        return self.cache[1]

    def gen_keys(self):
        for root_key in self.dataset.gen_keys():
            yield from self._augment(root_key).keys()

    def root_key(self, key):
        return key[0]

    def get_item(self, key):
        root_key = self.root_key(key)
        return self._augment(root_key)[key]

    @abstractmethod
    def augment(self, root_key, root_item):
        pass


class FilteredDataset(AugmentedDataset):
    def __init__(self, dataset, predicate, keep_positive=True):
        super().__init__(dataset)
        self.predicate = predicate
        self.keep_positive = keep_positive

    def augment(self, key, item):
        truth_value = self.predicate(key, item)
        if truth_value is self.keep_positive:
            yield (key, item)
        else:
            assert truth_value is (not self.keep_positive), (
                "Predicate {!r} should return a boolean value"
                .format(self.predicate)
            )

    def root_key(self, key):
        return key


class CachedDataset(Dataset):
    """Creates a dataset caching the result of another"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def _unforgotten_gen_keys(self):
        return self.dataset.gen_keys()
    gen_keys = _unforgotten_gen_keys

    def get_item(self, key):
        tup = self.cache.get(key, None)
        if tup is not None:
            return tup
        tup = self.dataset.get_item(key)
        self.cache[key] = tup
        return tup

    def _cached_keys(self):
        return self.cache.keys()

    def fill_forget(self):
        for key in self.dataset.gen_keys():
            self.get_item(key)
        self.gen_keys = self._cached_keys
        self.dataset = None
        return self


class PickledDataset(Dataset):
    """A dataset compacted on the disk with Pickle. For initial creation from
    an old dataset::

        in_mem_dataset = DictDataset({"a": 1, "b": 2, "c": 3})
        with open("file_path", "wb") as f:
            PickledDataset.create(in_mem_dataset, f)

    For using a PickledDataset::

        with open("file_path", "rb") as f:
            pd = PickledDataset(f)
            pd = TransformedDataset(pd, [lambda x, draw: (x, x)])
            X, Y = pd.query(pd.gen_keys())
            model.fit(X, Y)
    """
    @staticmethod
    def create(dataset, file_handler, gen_keys_wrapper=None, keys=None):
        if isinstance(file_handler, str):
            with open(file_handler, "wb") as file_handler:
                return PickledDataset.create(dataset, file_handler, keys=keys)
        index = {}
        pickler = Pickler(file_handler)
        # allocate space for index offset
        file_handler.seek(0)
        pickler.dump(1 << 65)  # 64 bits placeholder
        if keys is None:
            keys = dataset.gen_keys()
        if gen_keys_wrapper is not None:
            keys = gen_keys_wrapper(keys)
        for key in keys:
            # pickle objects and build index
            index[key] = file_handler.tell()
            obj = dataset.get_item(key)
            pickler.dump(obj)
            pickler.memo.clear()
        # put index and record offset
        index_location = file_handler.tell()
        pickler.dump(index)
        # put context
        context = getattr(dataset, "_context", None)
        if context:
            pickler.dump({**context})
        # put index offset at the beginning of the file
        file_handler.seek(0)
        index_location ^= 1 << 65
        pickler.dump(index_location)

    def __init__(self, file_handler):
        if isinstance(file_handler, str):
            file_handler = open(file_handler, "rb")
        self.file_handler = file_handler
        self.unpickler = unpickler = Unpickler(file_handler)

        # load the index offset then the index
        file_handler.seek(0)
        index_location = unpickler.load()
        index_location ^= 1 << 65
        file_handler.seek(index_location)
        self.index = unpickler.load()
        # try to load the context if any
        try:
            self._context = ChainMap(unpickler.load())
        except EOFError:
            pass

    def __getstate__(self):
        return (self.file_handler.name,)

    def __setstate__(self, state):
        self.__init__(*state)

    def gen_keys(self):
        return self.index.keys()

    def get_item(self, key):
        self.file_handler.seek(self.index[key])
        ret = self.unpickler.load()
        self.unpickler.memo.clear()
        return ret


class DiffReason(Exception):
    pass


def _recursive_equality(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        for a_, b_ in zip(a, b):
            if not _recursive_equality(a_, b_):
                return False
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        keys = a.keys()
        if keys != b.keys():
            return False
        for key in keys:
            if not _recursive_equality(a[key], b[key]):
                return False
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def pickle_or_load(dataset, path, keys=None, *, check_first_n_items=1, before_pickling=None,
                   gen_keys_wrapper=None, are_equal=_recursive_equality):
    from io import IOBase
    if isinstance(path, IOBase):
        PickledDataset.create(dataset, path, keys=keys, gen_keys_wrapper=gen_keys_wrapper)
        return PickledDataset(path)
    was_existing = os.path.exists(path)
    if not was_existing:
        file = None
        if before_pickling is not None:
            before_pickling()
        try:
            with open(path, "wb") as file:
                PickledDataset.create(dataset, file, keys=keys, gen_keys_wrapper=gen_keys_wrapper)
        except BaseException as exc:  # catch ALL exceptions
            if file is not None:  # if the file has been created, it is partial
                os.remove(path)
            raise
    opened_dataset = PickledDataset(open(path, "rb"))
    if keys is None:
        keys = dataset.gen_keys()
    chunk = next(chunkify(keys, check_first_n_items))
    reason = None
    for k in chunk:
        true_item = dataset.get_item(k)
        try:
            try:
                loaded_item = opened_dataset.get_item(k)
            except KeyError as exc:
                raise DiffReason("Pickled dataset does not contain key " +
                                 str(exc))
            equality = are_equal(true_item, loaded_item)
        except DiffReason as r:
            reason = r
            equality = False
        except Exception:
            print("Warning: Could not check whether the dataset pickled at {} "
                  "was up to date.\n".format(path),
                  file=sys.stderr)
            raise
        if not equality:
            if reason is not None:
                print("Warning: Pickled dataset at {} seems to be out of date."
                      "\nReason: {}".format(path, str(reason),
                      file=sys.stderr))
            else:
                print("Warning: Pickled dataset at {} seems to be out of date. "
                      "Or are_equal may be wrongly implemented.".format(path),
                      file=sys.stderr)
            if not was_existing:
                print("Since the dataset has just been created, it you may want "
                      "to check the determinism of dataset.get_item.",
                      file=sys.stderr)
            break
    return opened_dataset


try:
    import blosc
except ImportError:
    pass


class SqueezedDataset(Dataset):
    config = (9, "blosclz", True)

    def __init__(self, dataset, keys, config=None):
        self.dataset = dataset
        self.keys = keys
        if config is not None:
            self.config = config

    def gen_keys(self):
        return self.dataset.gen_keys()

    def get_item(self, key):
        clevel, cname, shuffle = self.config

        item = self.dataset.get_item(key)
        for key in self.keys:
            array = item[key]
            array = np.ascontiguousarray(array)
            shape, size, dtype = array.shape, array.size, array.dtype
            comp = blosc.compress_ptr(array.__array_interface__['data'][0], size,
                                      typesize=dtype.itemsize, clevel=clevel,
                                      cname=cname, shuffle=shuffle)
            item[key] = _SqueezedArray(shape, dtype, comp)
        return item


class ExpandedDataset(Dataset):
    def __init__(self, dataset, keys, config=None):
        self.dataset = dataset
        self.keys = frozenset(keys)
        if config is not None:
            self.config = config

    def gen_keys(self):
        return self.dataset.gen_keys()

    def get_item(self, key):
        item = self.dataset.get_item(key)
        for key in tuple(item.keys()):
            value = item[key]
            if isinstance(value, _SqueezedArray):
                if key not in self.keys:
                    del item[key]
                    continue
                shape, dtype, comp = item[key]
                array = np.empty(shape, dtype=dtype)
                blosc.decompress_ptr(comp, array.__array_interface__['data'][0])
                item[key] = array
        return item


_SqueezedArray = namedtuple("_SqueezedArray", "shape, dtype, comp")


class DictDataset(Dataset):
    """Mostly an example for a simple in-memory dataset"""
    def __init__(self, dic):
        self.dic = dic

    def gen_keys(self):
        return self.dic.keys()

    def get_item(self, key):
        return self.dic[key]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
