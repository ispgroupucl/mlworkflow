import pytest
from mlworkflow import djson_dumps, djsonc_loads


def test_copying():
    x = djsonc_loads('''
    {
        "options": {
            "lr": [1e-3, 1e-4, 1e-5],
            "scale": 4,
            "optimizer": {
                "name": "Adam",
                "lr": {"_tuple": [{"_copy": "options.lr.1"}]}
            }
        },
        "dataset": {
            "scale": {"_copy": "options.scale"}
        }
    }
    ''')
    assert x["dataset"]["scale"] == 4
    assert x["options"]["optimizer"]["lr"] == (1e-4,)


def test_load_dump():
    x = {0: "a", 1: "b"}
    assert djson_dumps(x, separators=(', ',': ')) == '{"_dict": [[0, "a"], [1, "b"]]}'
    assert djsonc_loads(djson_dumps(x)) == x


    x = {(): "a", 1: "b"}
    assert djsonc_loads(djson_dumps(x)) == x

    x = {((),):""}
    assert djson_dumps(x) == '{"_dict":[[{"_tuple":[{"_tuple":[]}]},""]]}'
    assert djsonc_loads(djson_dumps(x)) == x


def test_comments():
    assert djsonc_loads('''
        // Some comment
        [{
            "0": {"_tuple": /* intra-comment */ [1]}, // Some comment
            /* inter
               comment */
            "1": {"_tuple":[2]}, // Some comment
            // Some comment
        },
        // Some other comment
        ]
        // Some comment
    ''') == [{"0": (1,), "1": (2,)}]

    import json
    with pytest.raises(json.JSONDecodeError):
        djsonc_loads('''[],''')
