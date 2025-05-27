# src/graphs/graph_repository.py

from typing import Literal, Dict, Callable
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.graphs.base_graph import BaseGraph

GraphType = Literal["main"]


class GraphRepository:
    """
    Repository Pattern: Tüm graph yapılarını merkezi olarak yönetir.
    Her graph sınıfı BaseGraph'den türemeli ve llm ile initialize edilmelidir.
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
        Belirtilen graph türünü üretir ve döner.
        """
        if graph_type not in self._registry:
            raise ValueError(f"Geçersiz graph tipi: {graph_type}")
        return self._registry[graph_type]()

    def list_graphs(self):
        """
        Desteklenen tüm graph türlerini listeler.
        """
        return list(self._registry.keys())

    def register(self, graph_type: GraphType, builder: Callable[[], BaseGraph]):
        """
        Dışarıdan yeni bir graph tipi kaydeder (test, plugin, override senaryoları için).
        """
        self._registry[graph_type] = builder
