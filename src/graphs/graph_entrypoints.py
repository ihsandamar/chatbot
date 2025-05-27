from src.graphs.graph_repository import GraphRepository
from src.models import LLM
from src.graphs.main_graph import MainGraph


def graph(name: str) -> MainGraph:
    """
    Belirtilen graph adını kullanarak graph nesnesi döner.
    """
    from src.config import OPENAI_API_KEY
    llm = LLM(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    repo = GraphRepository(llm=llm)
    return repo.get(name)

def main_graph():
    return graph("main").build_graph()

