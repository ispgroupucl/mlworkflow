from .misc import DictObject

import importlib
import textwrap
import warnings


class MetaLazy(type):
    def __new__(cls, name, bases, dic):
        lazy_fields = {}
        for base in bases:
            lazy_fields.update(getattr(base, "_MetaLazy__lazy_fields", ()))

        for factory in dic:
            if factory.endswith("__init"):
                pname = factory[:-6]
                lazy_fields[pname] = factory
        dic["_MetaLazy__lazy_fields"] = lazy_fields

        return super().__new__(cls, name, bases, dic)


class Lazy(metaclass=MetaLazy):
    def __getattr__(self, name):
        factory = self._MetaLazy__lazy_fields.get(name, None)
        if factory is not None:
            res = getattr(self, factory)()
            setattr(self, name, res)
            return res
        return getattr(super(), name)


_no_value = object()
class LazyConfigurable(Lazy):
    def __init__(self, cfg):
        self.cfg = cfg

    def __getitem__(self, key):
        current = self.cfg
        for k in key.split("."):
            current = current[k]
        return current

    def __setitem__(self, key, value):
        current = self.cfg
        *parents, last_k = key.split(".")
        for k in parents:
            d = current.get(k, _no_value)
            if d is _no_value:
                current[k] = {}
            current = current[k]
        current[last_k] = value


def get_callable(name):
    module, *fun = name.split(" ", 1)
    fun = module.split(".")[-1] if not fun else fun[0]

    current = importlib.import_module(module)
    for n in fun.split("."):
        current = getattr(current, n)
    return current


def cfg_call(d, *args, **kwargs):
    if isinstance(d, str):
        return get_callable(d)(*args, **kwargs)
    callee = get_callable(d.pop("_"))
    return callee(*args, **kwargs, **d)


def exec_dict(dst, source, env):
    for qualname, exp in source.items():
        current = dst
        *parents, name = qualname.split(".")
        for parname in parents:
            if parname not in current:
                current[parname] = DictObject()
                if current is dst:
                    env[parname] = current[parname]
            current = current[parname]

        try:
            current[name] = eval(exp, env)
            if current is dst:
                env[name] = current[name]
        except:
            raise Exception("Error evaluating\n"+textwrap.indent(exp, prefix="    "))
    return dst


def exec_flat(dst, source, env):
    for name, exp in source.items():
        dst[name] = eval(exp, env)
    return dst


def flat_to_dict(dst, flat):
    for qualname, value in flat.items():
        *parents, name = qualname.split(".")
        current = dst
        for parname in parents:
            if parname not in current:
                current[parname] = DictObject()
            current = current[parname]
        current[name] = value
    return dst
