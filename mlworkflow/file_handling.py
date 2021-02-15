import glob
import os


def join_base(path, base=""):
    if base.endswith(".py"):
        base = os.path.dirname(base)
    return os.path.join(base, path)


def find_files(filenames, base="", recursive=True):
    if base.endswith(".py"):
        base = os.path.dirname(base)
    if not base:
        base = "."
    _added_slash = not base.endswith(os.sep)

    if not isinstance(filenames, list):
        filenames = [filenames]
    found = set()
    for filename in filenames:
        found.update(glob.glob(os.path.join(base, filename), recursive=recursive))
    lst = [fn[len(base)+_added_slash:] for fn in found]
    lst.sort()
    return lst
