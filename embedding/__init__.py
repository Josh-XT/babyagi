import importlib

class Embedding:
    def __init__(self, name):
        try:
            module = importlib.import_module(f".{name}", package=__name__)
            embedding_class = getattr(module, f"{name.capitalize()}Embedding")
            self.instance = embedding_class()
        except (ModuleNotFoundError, AttributeError) as e:
            raise AttributeError(f"module {__name__} has no attribute {name}") from e

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

def __getattr__(name):
    return Embedding(name)
