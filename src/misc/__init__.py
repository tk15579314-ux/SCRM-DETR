from .logger import *


def __getattr__(name):
    if name == 'show_sample':
        from .visualizer import show_sample
        return show_sample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
