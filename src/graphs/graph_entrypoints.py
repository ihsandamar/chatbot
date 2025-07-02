from src.graphs.graph_repository import GraphRepository
from src.models import LLM
from src.graphs.main_graph import MainGraph
from src.services.config_loader import ConfigLoader


# TODO: Add Dynamic Module Discovery and Loading  

def graph(name: str) -> MainGraph:
    """
    Returns a graph object based on the specified name.
    
    :param name: Name of the graph
    :return: Graph object
    :rtype: MainGraph
    """

    config = ConfigLoader.load_config()
    OPENAI_API_KEY = config["llm"]["api_key"]
    
    llm = LLM(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    repo = GraphRepository(llm=llm)
    return repo.get_raw(name)

def main_graph():
    return graph("main").build_graph()


def text2sql_graph():
    return graph("text2sql").build_graph()


def main_graph_with_tools():
    return graph("main_graph_with_tools").build_graph()

