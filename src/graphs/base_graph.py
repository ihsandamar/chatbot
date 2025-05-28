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
        Each subclass must override this function to return its own StateGraph structure.
        """
        pass

    def get_graph_builder(self) -> StateGraph:
        """
        Provides the graph builder (StateGraph) object.
        Subclasses can override if needed.
        """
        return StateGraph(self.state_class)

    def get_memory(self):
        """
        Returns the memory object used by the graph.
        """
        return self.memory
