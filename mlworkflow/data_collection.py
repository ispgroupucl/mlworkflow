from .data_freezing import (ImageSaver, Pickleb64Freezer,
    PickleSaver)
from .file_handling import join_base
from .json_handling import DJSON
from .misc import gen_id

from collections import ChainMap, defaultdict
from functools import wraps
import json
import os


class _Provider:
    image = png = property(ImageSaver)
    pickle = property(PickleSaver)
    pickleb64 = property(Pickleb64Freezer)

    @property
    def dirname(self):
        return os.path.dirname(self.filename)

    def add_metadata(self, dic):
        filename = self.filename if isinstance(self, _Provider) else self
        assert isinstance(dic, dict), ("metadata must take the form of a "
                                       "dictionary")
        with open(filename+"_", "a") as file:
            json.dump(DJSON.to_json(dic), file, separators=(',',':'))
            file.write("\n")
            file.flush()

    def get_metadata(self):
        filename = self.filename if isinstance(self, _Provider) else self
        metadata = {}
        try:
            with open(filename+"_", "r") as file:
                for obj in DataCollection._read_json(file):
                    metadata.update(obj)
        except FileNotFoundError:
            pass
        return metadata


class DataCollection(ChainMap, _Provider):
    """A class for recording experimental results
    """

    @staticmethod
    def _read_json(file):
        while True:
            s = file.readline()
            if not s:
                break
            yield DJSON.from_json(json.loads(s))

    @staticmethod
    def load_file(filename, base=""):
        filename = join_base(filename, base)
        with open(filename, "r") as file:
            return DataCollection._load_file_from_fp(file, filename)

    @staticmethod
    def _load_file_from_fp(file, filename):
        cum = {}
        data = []
        for obj in DataCollection._read_json(file):
            cum = {**cum, **obj}  # Cumulate fields
            data.append(_CheckPointWrapper(cum))  # Wrap
        return _CheckPointFileWrapper(data, filename=filename)

    def __init__(self, filename="{}.json", append=False):
        self._sparse = {}
        self._cumulated = {}
        super().__init__(self._sparse, self._cumulated)

        self.filename = gen_id(filename)
        if os.path.exists(self.filename):
            assert append, ("{} already exists, append option is necessary to continue"
                            .format(self.filename))
            self.file = open(self.filename, "r+")
            self.history = DataCollection._load_file_from_fp(self.file, self.filename)
            self._cumulated.update(self.history[-1])
        else:
            self.file = open(self.filename, "w")
            self.history = _CheckPointFileWrapper([], filename=self.filename)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.history.__getitem__(key)
        if isinstance(key, list):
            sup = super()
            return [sup.__getitem__(k) for k in key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            assert len(key) == len(value)
            for k, v in zip(key, value):
                assert isinstance(k, str)
                super().__setitem__(k, v)
        else:
            assert isinstance(key, str)
            super().__setitem__(key, value)

    @property
    def iteration(self):
        return len(self.history)

    @property
    def history_(self):
        return _CheckPointFileWrapper(self.history+[_CheckPointWrapper({**self._cumulated, **self._sparse})],
                                      filename=self.filename)

    def checkpoint(self):
        sparse = self._sparse
        cumulated = self._cumulated
        # Write sparse to file
        json.dump(DJSON.to_json(sparse), self.file, separators=(',',':'))
        self.file.write("\n")
        self.file.flush()
        # Update cumulated and history with a frozen version
        cumulated.update(sparse)
        frozen = cumulated.copy()
        self.history.append(_CheckPointWrapper(frozen))
        sparse.clear()


class _CheckPointWrapper(dict):
    """"Add multiple and optional selections for a dict """
    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().get(key.start, key.stop)
        if isinstance(key, list):
            sup = super()
            return [sup.__getitem__(k) for k in key]
        return super().__getitem__(key)


class _CheckPointFileWrapper(list, _Provider):
    """Add slice selection for a list of CheckPointWrapper"""
    def __init__(self, *args, filename):
        super().__init__(*args)
        self.filename = filename

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2, ("Key tuple must be of length 2,"
                                   "got {!r}".format(key))
            key0 = key[0]
            key1 = key[1]
            if isinstance(key0, slice):
                sup = super()
                return [l[key1] for l in sup.__getitem__(key0)]
            return super().__getitem__(key0)[key1]
        return super().__getitem__(key)

