import dis
import sys
import re


_unpacking_code = dis.opmap["UNPACK_SEQUENCE"]
_map_from_op = {
    dis.opmap["STORE_NAME"]: "co_names",
    dis.opmap["STORE_GLOBAL"]: "co_names",
    dis.opmap["STORE_FAST"]: "co_varnames",
}


class Unpackable:
    def __upck__(self, name):
        raise NotImplementedError()
    
    def __iter__(self):
        upper = sys._getframe(1)
        offset = upper.f_lasti
        code = upper.f_code.co_code
        if code[offset] == _unpacking_code:
            n = code[offset+1]
            iterees = []
            offset += 2
            for i in range(n):
                name_cat = _map_from_op[code[offset]]
                name = getattr(upper.f_code, name_cat)[code[offset+1]]
                iterees.append(self.__upck__(name))
                offset += 2
            return iter(iterees)
        return iter(super())


_default = object()
class ItemUnpackable(Unpackable):
    def __init__(self, obj, default=_default):
        self.obj = obj
        self.default = default
    def __upck__(self, name):
        if self.default is _default:
            return self.obj[name]
        else:
            return self.obj.get(name, self.default)


class AttrUnpackage(Unpackable):
    def __init__(self, obj, default=_default):
        self.obj = obj
        self.default = default
    def __upck__(self, name):
        if self.default is _default:
            return getattr(self.obj, name)
        else:
            return getattr(self.obj, name, self.default)


def unpack(obj, mode="item", default=_default):
    cls = {"item": ItemUnpackable, "attr": AttrUnpackage}[mode]
    return cls(obj, default=default)


_unpack_cache = {}
capturer = re.compile(r", *")
def _unpack(obj, keys, mode, default):
    _id = (keys, mode, default is _default)
    unpacker = _unpack_cache.get(_id, None)
    if unpacker is None:
        keys = capturer.split(keys)
        if mode == "item":
            formatter = "obj.get({!r},default)"
        elif mode == "attr":
            formatter = "getattr(obj,{!r},default)"
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        if default is _default:
            formatter = formatter.replace(",default", "")
        
        code = "lambda obj,default: ({},)".format(
               ",".join([formatter.format(key) for key in keys])
               )
            
        unpacker = eval(code, dict())
        _unpack_cache[_id] = unpacker
    return unpacker(obj, default)


def unpitems(obj, keys, default=_default):
    return _unpack(obj, keys, "item", _default)


def unpattrs(obj, keys, default=_default):
    return _unpack(obj, keys, "attr", _default)
