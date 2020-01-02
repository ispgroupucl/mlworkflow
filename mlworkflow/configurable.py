import importlib
import textwrap

from .misc import DictObject


class lazyproperty:
    """Declares a property to be lazy (evaluated only if absent from the object)
    """
    def __init__(self, initializer):
        self.initializer = initializer
    def __get__(self, instance, ownerclass=None):
        assert False, "lazyproperty should be used within subclasses of Lazy only."
        return self.initializer(instance)


_NOVALUE = object()
class _lazyproperty:
    """A placeholder here solely for the purpose of replacing potential class attributes
    that could otherwise propagate by inheritance.

    # Technical subtleties
    https://docs.python.org/3/howto/descriptor.html#descriptor-protocol
    This placeholder is a non-data descriptor, **which is of very high importance**.
    This allows all the logic behind them to be customizable by the host class
    through __getattr__, while providing __dict__ the precedence over it.

    Without that, we could not be able to handle elegantly __getattr__ for both
    lazy properties and non lazy attributes. Had we __initlazy__, __setlazy__
    and __dellazy__, it would be disturbing that __initlazy__ comes before __getattr__,
    but __setlazy__ and __dellazy__ would have to be called by __setattr__ and
    __delattr__.
    I have tried to deal with it, but some confusing logic would have been to be
    duplicated, and only to reach poor performance.
    """
    def __init__(self, name):
        self.name = name
    def __get__(self, instance, tpe):
        # NOTE: __getattr__ is called twice if an AttributeError is raised
        # raising AttributeError is 50% slower, which is best ?
        # return instance.__getattr__(self.name)
        raise AttributeError


class MetaLazy(type):
    def __new__(cls, name, bases, dic):
        lazy_fields = {}
        for base in bases:
            lazy_fields.update(getattr(base, "_MetaLazy__lazy_fields", ()))

        for name, value in tuple(dic.items()):
            if isinstance(value, lazyproperty):
                dic[name] = _lazyproperty(name)
                lazy_fields[name] = value.initializer

        dic["_MetaLazy__lazy_fields"] = lazy_fields

        return super().__new__(cls, name, bases, dic)


class Lazy(metaclass=MetaLazy):
    def __getattr__(self, name):
        initializer = self._MetaLazy__lazy_fields.get(name, None)
        if initializer is not None:
            value = initializer(self)
            self.__class__.__setlazy__(self, name, value)
            return value
        __getattr__ = getattr(super(), "__getattr__", None)
        if __getattr__ is not None:
            return __getattr__(name)
        cls = self.__class__
        raise AttributeError("Attribute {!r} not found in {!r}.".format(name,
            cls.__module__+"."+cls.__qualname__))

    # Minimal overhead attribute setting
    # One can change it with __setlazy__ = setattr
    __setlazy__ = object.__setattr__


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
            d = current.get(k, _NOVALUE)
            if d is _NOVALUE:
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
    d = d.copy()
    callee = get_callable(d.pop("_"))
    return callee(*args, **kwargs, **d)


def exec_dict(dst, statements, env):
    for qualname, exp in statements:
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


def exec_flat(dst, statements, env):
    for name, exp in statements:
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
