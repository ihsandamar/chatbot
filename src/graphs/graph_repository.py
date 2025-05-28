# src/graphs/graph_repository.py

from typing import Literal, Dict, Callable
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.graphs.base_graph import BaseGraph

GraphType = Literal["main"]


class GraphRepository:
    """
    GraphRepository manages all graph structures centrally using the Repository Pattern.
    Each graph class should inherit from BaseGraph and be initialized with an LLM instance.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self._registry: Dict[GraphType, Callable[[], BaseGraph]] = {
            "main": lambda: MainGraph(llm=self.llm),
            # İleride eklenebilir:
            # "text2sql": lambda: Text2SQLGraph(llm=self.llm),
            # "rag": lambda: RAGGraph(llm=self.llm),
        }

    def get(self, graph_type: GraphType) -> BaseGraph:
        """
        Produces and returns the specified graph type.
        """
        if graph_type not in self._registry:
            raise ValueError(f"Geçersiz graph tipi: {graph_type}")
        return self._registry[graph_type]()

    def list_graphs(self):
        """
        Lists all supported graph types.
        """
        return list(self._registry.keys())

    def register(self, graph_type: GraphType, builder: Callable[[], BaseGraph]):
        """
        Registers a new graph type from the outside (for testing, plugin, override scenarios).
        """
        self._registry[graph_type] = builder
