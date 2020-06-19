import pytest
from mlworkflow import lazyproperty, LazyConfigurable

class A(LazyConfigurable):
    @lazyproperty
    def a(self):
        return "Aa"

    @lazyproperty
    def b(self):
        return object()

class B(LazyConfigurable):
    @lazyproperty
    def a(self):
        return "Ba"

    @lazyproperty
    def b(self):
        return object()

    @lazyproperty
    def c(self):
        return "Bc"

class C(A):
    @lazyproperty
    def c(self):
        return "Cc"

def test_cache():
    inst = A({})
    assert inst.a == "Aa"

    cls = type("Something", (A,), {})
    inst = cls({})
    assert inst.a == "Aa"
    _1 = inst.b
    _2 = inst.b
    assert _1 is _2


def test_mro():
    inst = B({})
    assert inst.a == "Ba"

    cls = type("Something", (B, A,), {})
    inst = cls({})
    assert inst.a == "Ba"

    cls = type("Something", (C, B, A,), {})
    inst = cls({})
    assert inst.a == "Ba"
    