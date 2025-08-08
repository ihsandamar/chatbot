from langgraph_supervisor import create_supervisor

from src.graphs.chat_graph import ChatGraph
from src.graphs.registry import register_graph
from src.graphs.text2sql_graph import Text2SQLGraph
from langchain_core.tools import tool
from datetime import datetime  
from src.graphs.base_graph import BaseGraph

from langgraph.graph import StateGraph, START, END
from src.graphs.base_graph import BaseGraph
from src.models.models import LLM, State
from langgraph.checkpoint.memory import MemorySaver
@tool
def get_today() -> str:
    "Return today"
    return datetime.today().strftime('%Y-%m-%d')



@register_graph("supervisor")
class SupervisorTestGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm, State)





    def build_graph(self):
        chat_graph = ChatGraph(self.llm).build_graph()
        # Ensure text2sql_graph has a name for supervisor
        text2sql_graph_instance = Text2SQLGraph(self.llm)
        text2sql_graph = text2sql_graph_instance.build_graph()
        supervisor = create_supervisor(
            model=self.llm.get_chat(), 
            agents=[chat_graph, text2sql_graph], 
            tools=[get_today],
            prompt=(
                "You are a team supervisor managing a research expert and a math expert. "
                "For general prompt, use chat_graph. "
                "For sql problems, use text2sql_graph."
            )
        )
        memory = MemorySaver()
        return supervisor.compile(checkpointer=memory)                