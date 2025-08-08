from src.graphs.graph_repository import GraphRepository
from src.graphs.text2sql_graph import Text2SQLGraph
from src.models.models import LLM
from src.graphs.chat_graph import ChatGraph
from src.services.config_loader import ConfigLoader
from langchain_community.utilities import SQLDatabase

# TODO: Add Dynamic Module Discovery and Loading  

def graph(name: str) -> ChatGraph:
    """
    Returns a graph object based on the specified name.
    
    :param name: Name of the graph
    :return: Graph object
    :rtype: MainGraph
    """

    config = ConfigLoader.load_config()
    OPENAI_API_KEY = config.llm.api_key
    
    llm = LLM(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    repo = GraphRepository(llm=llm)
    return repo.get_raw(name)


def text2sql_graph():
    config = ConfigLoader.load_config("config/text2sql_config.yaml")
    llm = LLM(model=config.llm.model, temperature=config.llm.temperature, api_key=config.llm.api_key)
    db = SQLDatabase.from_uri(config.database.uri)
    graph = Text2SQLGraph(llm=llm, db=db).build_graph()
    return graph




def supervisor_graph():
    return graph("supervisor").build_graph()

