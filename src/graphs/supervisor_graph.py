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
        # Use text2sql_graph instance as agent (not the built graph)
        text2sql_graph_agent = Text2SQLGraph(self.llm).build_graph()
        supervisor = create_supervisor(
            model=self.llm.get_chat(), 
            agents=[chat_graph, text2sql_graph_agent], 
            tools=[get_today],
            prompt=(
                "You are an intelligent customer service supervisor managing specialized support agents. "
                "Your role is to analyze customer requests and direct them to the most appropriate expert. "
                "\n"
                "Available specialists:\n"
                "- chat_graph: General customer support specialist for conversations, questions, explanations, and assistance\n"
                "- text2sql_graph: Database query specialist for converting customer requests into SQL queries and retrieving data\n"
                "\n"
                "Decision criteria:\n"
                "- Use chat_graph for: General conversations, product information, troubleshooting, explanations, customer service inquiries\n"
                "- Use text2sql_graph for: Data retrieval requests, report generation, database queries, when customer asks to 'get data from', 'show me records', 'query database', 'retrieve information from tables'\n"
                "\n"
                "Always provide helpful, professional customer service. Be proactive in understanding customer needs."
            )
        )
        memory = MemorySaver()
        return supervisor.compile(checkpointer=memory)                