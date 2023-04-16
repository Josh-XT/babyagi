import importlib

class VectorDB:
    def __init__(self, name):
        try:
            module = importlib.import_module(f".{name}", package=__name__)
            vectordb_class = getattr(module, f"{name.capitalize()}VectorDB")
            self.instance = vectordb_class()
        except (ModuleNotFoundError, AttributeError) as e:
            raise AttributeError(f"module {__name__} has no attribute {name}") from e

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

def __getattr__(name):
    return VectorDB(name)
