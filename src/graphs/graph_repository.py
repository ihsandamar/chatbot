from src.graphs.registry import GRAPH_REGISTRY
from src.models import LLM

class GraphRepository:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.register_all()  # Register all graphs on initialization

    def get(self, graph_type: str):
        if graph_type not in GRAPH_REGISTRY:
            raise ValueError(f"Graph tipi bulunamadı: {graph_type}")
        return GRAPH_REGISTRY[graph_type](self.llm).build_graph()

    def get_raw(self, graph_type: str):
        if graph_type not in GRAPH_REGISTRY:
            raise ValueError(f"Graph tipi bulunamadı: {graph_type}")
        return GRAPH_REGISTRY[graph_type](self.llm)

    def list_graphs(self):
        return list(GRAPH_REGISTRY.keys())
    
    def register_all(self):
        """
        Scans all Python modules under `src.graphs.*` and registers graphs defined with the decorator.
        """
        import pkgutil
        import importlib
        import src.graphs  
        package = src.graphs.__path__

        for _, module_name, _ in pkgutil.iter_modules(package):
            importlib.import_module(f"src.graphs.{module_name}")
    

