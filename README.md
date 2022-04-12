# MLWorkFlow

This library is a set of tools made to help machine learning workflows.

## Installation
Package is released on PyPI for convenient installation:
```bash
pip install mlworkflow
```

## Dataset API

Datasets can be structured as (key, value) pairs where the keys are light and allow efficient querying of the dataset while the values contain the heavy data. This is the idea on which builds the abstract class `mlworkflow.Dataset`. The dataset objects comes with multiple useful methods or properties:

- `Dataset.keys` is a generator of the dataset keys. Keys that need to be computed are computed once and stored for efficient reuse. `Dataset.keys.all()` returns a list of the dataset keys (requiring computing all of them).
- `Dataset.batches(batch_size, wrapper=np.ndarray)` is a generator yielding batches of `batch_size` from the dataset keys.
- `Dataset.__len__()` provides the number of pairs (key, value) stored in the dataset. When keys hasn’t been computed yet in the case of laizy chaining with other datasets, it raises an error.

The most basic way of browsing items of `dataset`, an instance of an `mlworkflow.Dataset` is:
```python
for key in dataset.keys:
    item = dataset.query_item(key)
```

### New dataset implementation

Every `mlworkflow.Dataset` object should implement the `yield_keys()` method, responsible to yield the dataset keys, and the `query_item(key)` method, responsible to compute the item corresponding to the given key. The key objects should be hashable, immutable python object (usually a `NamedTuple`) to prevent equality issues and allow using them as dictionary keys. The dataset items can be any python object. The simplest example of such dataset is made from a dictionary:

```python
>>> class DictDataset(Dataset):
...     def __init__(self, dic):
...         self.dic = dic
...     def yield_keys(self):
...         yield from self.dic.keys()
...     def query_item(self, key):
...         return self.dic[key]
>>> parent = DictDataset({1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 5: "f"})
```


### Useful generic datasets

The library comes with multiple useful additions to the basic dataset object.

#### Filtered Dataset

`mlworkflow.FilteredDataset(parent, predicate)` allows the filtering of a dataset. The predicate receives the pair (key, value) of the parent dataset and returns `True` for the pairs that should be kept in the child dataset.

```python
>>> dataset = FilteredDataset(parent, predicate=lambda k,v: k%2==0)
>>> [dataset.query_item(key) for key in dataset.yield_keys()]
["b", "d", "f"]
```

#### Augmented Dataset

`mlworkflow.AugmentedDataset(parent)` augments a dataset in the sense that it can produce many items from one item of the dataset. Such dataset must implement the augment method that yields zero, one or multiple item given a (key, value) pair of the parent dataset

```python
>>> class PermutingDataset(AugmentedDataset):
...     def augment(self, root_key, root_item):
...         yield (root_key, 0), root_item+"a"
...         yield (root_key, 1), root_item+"b"
>>> dataset = PermutingDataset(parent)
>>> [dataset.query_item(key) for key in dataset.yield_keys()]
["aa", "ab", "ba", "bb", "ca", "cb", "da", "db", "ea", "eb", "fa", "fb]
```

#### Transformed Dataset

`mlworkflow.TransformedDataset(parent, transforms)` apply a list of transformation to a dataset. Each transformation must implement a `__call__` method applied on each (key, value) pair of the parent dataset. The keys remain unchanged.

```python
>>> dataset = TransformedDataset(parent, [lambda k,v: v.upper()])
>>> d.query_item(3)
"C"
```

#### Pickled Dataset

`mlworkflow.PickledDataset.create(dataset, filename)` is a static method allowing to create a dataset file on disk using the pickle library. The keys and items are stored allowing efficient access to the data later.

`mlworkflow.PickledDataset(filename)` loads a Pickled dataset from disk.


## Miscallenous

### Side Runner

`SideRunner` is an object allowing to parallellize computation. The constructor takes the number of threads as an argument and the object implements the following methods:

- `do(f, *args, **kwargs)` runs function `f` with `*args` arguments and `**kwargs` keyword arguments. The function is ran without any hold on it’s completion.
- `run_async(f, *args, **kwargs)` runs function `f` with `*args` arguments and `**kwargs` keyword arguments. The function’s completion is caught with `collect_runs()`
- `yield_async(gen)` yield items from generator `gen` in another thread.

examples:
```python
side_runner = SideRunner()
def batch_generator(dataset, keys, batch_size):
    for keys, batch in dataset.batches(batch_size=batch_size, drop_incomplete=True):
        yield keys, batch
async_batch_generator = side_runner.yield_async(batch_generator)
```

```python
parallel_downloader = SideRunner(10)
for item in items:
    parallel_downloader.run_async(download_item, item)
parallel_downloader.collect_runs()
```
