# Global graph registry for dynamic graph registration
# src/graphs/registry.py

GRAPH_REGISTRY = {}

def register_graph(name):
    def decorator(cls):
        GRAPH_REGISTRY[name] = cls
        return cls
    return decorator
