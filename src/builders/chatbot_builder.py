# src/builders/chatbot_builder.py
from src.chatbot import Chatbot

class ChatbotBuilder:
    def __init__(self, container):
        self.container = container
        self._llm = None
        self._graph_type = "main"
        self._config = {"configurable": {"thread_id": "1"}}

    def with_model(self, model_name: str, temperature: float = 0.0, api_key: str = None):
        def llm_provider():
            from src.models import LLM
            return LLM(model=model_name, temperature=temperature, api_key=api_key)

        self.container.register("llm", llm_provider)
        self._llm = self.container.resolve("llm")
        return self

    def with_graph(self, graph_type: str):
        self._graph_type = graph_type
        def repo_provider():
            from src.graphs.graph_repository import GraphRepository
            return GraphRepository(self._llm)
        self.container.register("graph_repo", repo_provider)
        return self

    def with_config(self, config: dict):
        self._config = config
        return self

    def build(self):
        graph = self.container.resolve("graph_repo").get(self._graph_type)
        return Chatbot(llm=self._llm, config=self._config, graph=graph)
