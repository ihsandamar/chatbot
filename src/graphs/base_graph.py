from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import Type

from src.models import LLM

class BaseGraph(ABC):
    """
    BaseGraph bir arayüz görevi görür. Her alt sınıf,
    en azından build_graph() metodunu override etmelidir.
    """

    def __init__(self, llm: LLM, state_class: Type):
        self.llm = llm
        self.state_class = state_class
        self.memory = MemorySaver()  # default belleği burada tutar, istenirse override edilebilir

    @abstractmethod
    def build_graph(self):
        """
        Her alt sınıf bu fonksiyonu override edip kendi StateGraph yapısını döndürmelidir.
        """
        pass

    def get_graph_builder(self) -> StateGraph:
        """
        Graph builder (StateGraph) nesnesini sağlar.
        Alt sınıflar gerekirse override edebilir.
        """
        return StateGraph(self.state_class)

    def get_memory(self):
        """
        Varsayılan olarak MemorySaver kullanılır.
        """
        return self.memory
