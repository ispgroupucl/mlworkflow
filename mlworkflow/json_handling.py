"""A module to provide complex dicts and tuples through JSON which only
supports dicts with strings as keys and lists.

This format is however not convient to write manually.
You may rather want to write your configs using the standard library
configparser along with exec_dict, etc from mlworkflow.configurable.
"""


class DJSON:
    @staticmethod
    def from_json(json):
        if isinstance(json, dict):
            if len(json) == 1:
                k = next(iter(json))
                if k == "_dict":
                    return dict(json[k])
                if k == "_tuple":
                    return tuple(json[k])
            parsed = {}
            for k, v in json.items():
                parsed[k] = DJSON.from_json(v)
            return parsed
        if isinstance(json, list):
            return [DJSON.from_json(el) for el in json]
        return json

    @staticmethod
    def to_json(djson):
        if isinstance(djson, dict):
            if any(not isinstance(k, str) for k in djson):
                return {"_dict": [[DJSON.to_json(k), DJSON.to_json(v)]
                                  for k, v in djson.items()
                                  ]}
            else:
                transformed = {}
                for k, v in djson.items():
                    transformed[k] = DJSON.to_json(v)
                return transformed
        if isinstance(djson, list):
            return [DJSON.to_json(el) for el in djson]
        if isinstance(djson, tuple):
            return  {"_tuple": DJSON.to_json(list(djson))}
        return djson
