from mlworkflow import DataCollection

def test_base():
    import os
    if os.path.exists("base_dc"):
        os.remove("base_dc")

    # Write 10 epochs
    dc = DataCollection("base_dc")
    for i in range(0, 10):
        dc["i"] = i
        dc.checkpoint()
    assert dc == {"i": 9}
    dc.file.close()
    # Load as list, check length
    dc = DataCollection.load_file("base_dc")
    assert len(dc) == 10
    assert dc[:,"i"] == list(range(10))
    # Reload for append
    dc = DataCollection("base_dc", append=True)
    assert dc == {"i": 9}
    # Write 10 more epochs
    for i in range(10, 20):
        dc["i"] = i
        dc.checkpoint()
    assert dc == {"i": 19}
    # Reload as list, should see 20 epochs
    dc = DataCollection.load_file("base_dc")
    assert len(dc) == 20
    assert dc[:,"i"] == list(range(20))
    if os.path.exists("base_dc"):
        os.remove("base_dc")
