from abc import ABCMeta, abstractmethod
from collections import ChainMap, namedtuple
import collections
import os
from pickle import _Pickler as Pickler, _Unpickler as Unpickler
import sys
import threading
import warnings

import numpy as np



def chunkify(iterable, n, drop_incomplete=False):
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
    if last_n == n or not drop_incomplete:
        yield ret[:last_n]  # yield the incomplete subset ([] if i = -1)


_NOVALUE = object()
class _lazyattr:
    """Very specific to this place, understand that it can be dangerous before
    using it anywhere else:
    It fills the attribute whose name is that of the function, and recomputes
    the result everytime it is not shielded by that attribute.
    """
    def __init__(self, f):
        self.f = f
        self.name = f.__name__
    def __get__(self, instance, owner):
        value = self.f(instance)
        setattr(instance, self.name, value)
        return value


class _DatasetKeys:
    @staticmethod
    def _from_yield_keys(yielded_keys):
        if isinstance(yielded_keys, _DatasetKeys):
            return yielded_keys
        if isinstance(yielded_keys, collections.abc.Sized):
            return _CompleteDatasetKeys(yielded_keys)
        return _DatasetKeys(yielded_keys)

    def __init__(self, generator):
        self.keys = []
        self.generator = iter(generator)

    def __getitem__(self, i):
        try:
            while i >= len(self.keys):
                self.keys.append(next(self.generator))
        except StopIteration:
            self.keys = tuple(self.keys)
            del self.generator
            self.__class__ = _CompleteDatasetKeys
        return self.keys[i]

    def all(self):
        self.keys = (*self.keys, *self.generator)
        del self.generator
        self.__class__ = _CompleteDatasetKeys
        return self.keys

    def __repr__(self):
        return "DatasetKeys(incomplete, {!r})".format(self.keys)


class _CompleteDatasetKeys:
    def __init__(self, sized):
        self.keys = tuple(sized)

    def all(self):
        return self.keys

    def __getitem__(self, i):
        return self.keys.__getitem__(i)

    def __len__(self):
        return self.keys.__len__()

    def __repr__(self):
        return "DatasetKeys(complete, {!r})".format(self.keys)


def _to_dict(obj):
    to_dict = getattr(obj, "to_dict", _NOVALUE)
    if to_dict is not _NOVALUE:
        return to_dict()
    return {**obj}


def _from_dict(obj, dic):
    from_dict = getattr(obj, "from_dict", _NOVALUE)
    if from_dict is not _NOVALUE:
        return from_dict(dic)
    return type(obj)(**dic)

def batchify(items, wrapper=np.array):
    """Transforms a list of (key, value) items (dictionaries or
    dictionarizable objects) into a dictionary of (key, wrapped values)
    """
    values = {}
    for item in items:
        item = _to_dict(item)
        if not values:
            values = {k:[v] for k, v in item.items()}
        else:
            for k in values:
                values[k].append(item[k])
    for k in values:
        values[k] = wrapper(values[k])
    return values


class Dataset(metaclass=ABCMeta):
    """The base class for any dataset, provides the and batches methods from
    yield_keys() and query_item(key)

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d.query([0, 1])
    (array(['Denzel', 'Tom'], ...), array(['Washington', 'Hanks'], ...))
    >>> list(d.batches([0, 1], 1))
    [(array(['Denzel'], ...), array(['Washington'], ...),
     (array(['Tom'], ...), array(['Hanks'], ...))]

    We can see the latter provides two batches
    """

    @abstractmethod
    def yield_keys(self):
        raise NotImplementedError()

    @abstractmethod
    def query_item(self, key):
        """Returns a tuple for one item, typically (Xi, Yi), or (Xi,)
        """
        pass

    @_lazyattr
    def keys(self):
        return _DatasetKeys._from_yield_keys(self.yield_keys())

    def query(self, keys, wrapper=np.array):
        return batchify([self.query_item(key) for key in keys], wrapper=wrapper)

    def batches(self, keys, batch_size, wrapper=np.array, drop_incomplete=False):
        for key_chunk in chunkify(keys, n=batch_size, drop_incomplete=drop_incomplete):
            yield key_chunk, self.query(key_chunk, wrapper)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        if self._parent is not None:
            self._context = parent.context.new_child(self.context.maps[0])

    @property
    def context(self):
        try:
            return self._context
        except AttributeError:
            self._context = ChainMap()
            return self._context

    @property
    def dataset(self):
        warnings.warn("'dataset.dataset' is deprecated. Please use 'dataset.parent'")
        return self.parent

    @dataset.setter
    def dataset(self, parent):
        warnings.warn("'dataset.dataset' is deprecated. Please use 'dataset.parent'")
        self.parent = parent


class TransformedDataset(Dataset):
    """A dataset that passes yielded items through transforms

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = TransformedDataset(d, [lambda x: (x[0][:1]+".", x[1])])
    >>> d.query([0, 1])
    (array(['D.', 'T.'], ...), array(['Washington', 'Hanks'], ...))
    """
    def __init__(self, parent, transforms=()):
        """Creates a dataset performing operations for modifying another"""
        self.parent = parent
        self.transforms = list(transforms)

    def yield_keys(self):
        return self.parent.keys

    def query_item(self, key):
        item = self.parent.query_item(key)
        for transform in self.transforms:
            item = transform(key, item)
        return item

    def add_transform(self, transform):
        self.transforms.append(transform)
        return transform

    def add_transforms(self, transforms):
        self.transforms.extend(transforms)


class AugmentedDataset(Dataset, metaclass=ABCMeta):
    """ "Augments" a dataset in the sense that it can produce many items from
    one item of the dataset.

    >>> class PermutingDataset(AugmentedDataset):
    ...     def augment(self, root_key, root_item):
    ...         yield (root_key, 0), root_item
    ...         yield (root_key, 1), root_item[::-1]
    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = PermutingDataset(d)
    >>> new_keys = d.keys()
    >>> new_keys
    ((0, 0), (0, 1), (1, 0), (1, 1))
    >>> d.query(new_keys)
    (array(['Denzel', 'Washington', 'Tom', 'Hanks'], ...),
     array(['Washington', 'Denzel', 'Hanks', 'Tom'], ...))
    """
    def __init__(self, parent):
        self.parent = parent
        self.cache = (None, None)

    def _augment(self, root_key):
        cache = self.cache
        if cache[0] != root_key:
            root_item = self.parent.query_item(root_key)
            new_items = dict(self.augment(root_key, root_item))
            cache = self.cache = (root_key, new_items)
        return cache[1]

    def yield_keys(self):
        keys = []
        for root_key in self.parent.keys:
            new_keys = self._augment(root_key).keys()
            keys.extend(new_keys)
            yield from new_keys

    def root_key(self, key):
        return key[0]

    def query_item(self, key):
        root_key = self.root_key(key)
        return self._augment(root_key)[key]

    @abstractmethod
    def augment(self, root_key, root_item):
        pass


class CachedDataset(Dataset):
    """Creates a dataset caching the result of another"""
    def __init__(self, parent):
        self.parent = parent
        self.cache = {}

    def yield_keys(self):
        return self.parent.keys

    def query_item(self, key):
        tup = self.cache.get(key, None)
        if tup is not None:
            return tup
        tup = self.parent.query_item(key)
        self.cache[key] = tup
        return tup

    def fill(self):
        for key in self.parent.keys:
            self.query_item(key)
        return self


class DictDataset(Dataset):
    """Mostly an example for a simple in-memory dataset"""
    def __init__(self, dic):
        self.dic = dic

    def yield_keys(self):
        return self.dic.keys()

    def query_item(self, key):
        return self.dic[key]


class FilteredDataset(AugmentedDataset):
    def __init__(self, parent, predicate, keep_positive=True):
        super().__init__(parent)
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
            X, Y = pd.query(pd.keys())
            model.fit(X, Y)
    """
    @staticmethod
    def create(dataset, file_handler, yield_keys_wrapper=None, keys=None):
        if isinstance(file_handler, str):
            with open(file_handler, "wb") as file_handler:
                return PickledDataset.create(dataset, file_handler, yield_keys_wrapper=yield_keys_wrapper, keys=keys)
        index = {}
        pickler = Pickler(file_handler)
        # allocate space for index offset
        file_handler.seek(0)
        pickler.dump(1 << 65)  # 64 bits placeholder
        if keys is None:
            keys = dataset.keys
        if yield_keys_wrapper is not None:
            keys = yield_keys_wrapper(keys)
        for key in keys:
            # pickle objects and build index
            index[key] = file_handler.tell()
            obj = dataset.query_item(key)
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
        unpickler.memo.clear()
        # try to load the context if any
        try:
            self._context = ChainMap(unpickler.load())
            unpickler.memo.clear()
        except EOFError:
            pass

        self.lock = threading.Lock()

    def __getstate__(self):
        return (self.file_handler.name,)

    def __setstate__(self, state):
        self.__init__(*state)

    def yield_keys(self):
        return self.index.keys()

    def query_item(self, key):
        self.lock.acquire()
        self.file_handler.seek(self.index[key])
        ret = self.unpickler.load()
        self.unpickler.memo.clear()
        self.lock.release()
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
                   yield_keys_wrapper=None, are_equal=_recursive_equality):
    from io import IOBase
    if isinstance(path, IOBase):
        PickledDataset.create(dataset, path, keys=keys, yield_keys_wrapper=yield_keys_wrapper)
        return PickledDataset(path)
    was_existing = os.path.exists(path)
    if not was_existing:
        file = None
        if before_pickling is not None:
            before_pickling()
        try:
            with open(path, "wb") as file:
                PickledDataset.create(dataset, file, keys=keys, yield_keys_wrapper=yield_keys_wrapper)
        except BaseException as exc:  # catch ALL exceptions
            if file is not None:  # if the file has been created, it is partial
                os.remove(path)
            raise
    opened_dataset = PickledDataset(open(path, "rb"))
    if keys is None:
        keys = dataset.keys
    chunk = next(chunkify(keys, check_first_n_items))
    reason = None
    for k in chunk:
        true_item = dataset.query_item(k)
        try:
            try:
                loaded_item = opened_dataset.query_item(k)
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
                print("Since the dataset has just been created, you may want "
                      "to check the determinism of dataset.query_item.",
                      file=sys.stderr)
            break
    return opened_dataset


try:
    import blosc
except ImportError:
    pass


def _squeezed_copy(obj, clevel, cname, shuffle):
    """Compress arrays within dicts, tuples and lists, do not dig other objects for now
    """
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        shape, size, dtype = array.shape, array.size, array.dtype
        comp = blosc.compress_ptr(array.__array_interface__['data'][0], size,
                                  typesize=dtype.itemsize, clevel=clevel,
                                  cname=cname, shuffle=shuffle)
        return _SqueezedArray(shape, dtype, comp)
    tpe = type(obj)
    if tpe is tuple or tpe is list:
        return tpe(_squeezed_copy(el, clevel, cname, shuffle) for el in obj)
    if tpe is dict:
        return tpe((k, _squeezed_copy(v, clevel, cname, shuffle))
                   for k, v in obj.items())
    return obj


def _expanded_copy(obj):
    """Expand arrays within dicts, tuples and lists, do not dig other objects for now
    """
    if isinstance(obj, _SqueezedArray):
        shape, dtype, comp = obj
        array = np.empty(shape, dtype=dtype)
        blosc.decompress_ptr(comp, array.__array_interface__['data'][0])
        return array
    tpe = type(obj)
    if tpe is tuple or tpe is list:
        return tpe(_expanded_copy(el) for el in obj)
    if tpe is dict:
        return tpe((k, _expanded_copy(v))
                   for k, v in obj.items())
    return obj


class SqueezedDataset(Dataset):
    config = (9, "blosclz", True)
    def __init__(self, parent, compressed_keys, config=None):
        self.parent = parent
        self.compressed_keys = compressed_keys
        if config is not None:
            self.config = config

    def yield_keys(self):
        return self.parent.keys

    def query_item(self, key):
        clevel, cname, shuffle = self.config
        raw_item = self.parent.query_item(key)

        item = _to_dict(raw_item)
        for key in self.compressed_keys:
            item[key] = _squeezed_copy(item[key], clevel, cname, shuffle)

        item = _from_dict(raw_item, item)
        return item


class ExpandedDataset(Dataset):
    def __init__(self, parent, compressed_keys):
        self.parent = parent
        self.compressed_keys = compressed_keys

    def yield_keys(self):
        return self.parent.keys

    def query_item(self, key):
        raw_item = self.parent.query_item(key)
        item = _to_dict(raw_item)

        for key in self.compressed_keys:
            item[key] = _expanded_copy(item[key])

        item = _from_dict(raw_item, item)
        return item


_SqueezedArray = namedtuple("_SqueezedArray", "shape, dtype, comp")


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)

