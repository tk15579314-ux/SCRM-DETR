import importlib.util
from importlib import import_module

_LAZY_SUBMODULES = {"nn", "optim", "zoo"}

if importlib.util.find_spec(f"{__name__}.data") is None:
    data = None
else:
    from . import data

__all__ = sorted(_LAZY_SUBMODULES)
if data is not None:
    __all__.append("data")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
