import json
from functools import wraps


def singleton_arg(func):
    _instances = {}

    @wraps(func)
    def wrapper_call(*args, **kwargs):
        arg_str = '%s_%s' % (json.dumps(args), json.dumps(kwargs))
        if arg_str not in _instances:
            ret = func(*args, **kwargs)
            _instances[arg_str] = ret
        return _instances[arg_str]

    return wrapper_call
