import importlib

def __getattr__(name):
    try:
        module = importlib.import_module(f".{name}", package=__name__)
        return module.instruct
    except (ModuleNotFoundError, AttributeError):
        raise AttributeError(f"module {__name__} has no attribute {name}")
