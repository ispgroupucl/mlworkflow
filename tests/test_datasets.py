import numpy as np
from mlworkflow.datasets import batchify

def test_batchify():
    chunk = [ { "a": 1, "b": 10}, { "a": 2, "b": 11}, { "a": 3, "b": 13} ]
    batch = batchify(chunk)
    assert isinstance(batch["a"], np.ndarray)
    assert len(batch["a"]) == 3
    assert batch["a"][1] == 2
    assert batch["b"][2] == 13