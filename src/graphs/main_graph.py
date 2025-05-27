from langgraph.graph import StateGraph, START, END
from src.graphs.base_graph import BaseGraph
from src.models import LLM, State
from langgraph.checkpoint.memory import MemorySaver



class MainGraph(BaseGraph):
    """
    Main Graph for the chatbot workflow.
    This graph defines the structure and flow of the chatbot interactions.
    """
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=State)

    def chatbot(self, state: State):
        return {"messages": [self.llm.send(state["messages"])]}


    def build_graph(self):
        """
        Function to build the graph.
        This is useful for testing purposes.
        """
        memory = MemorySaver()
        graph_builder = StateGraph(State)

        # Create the graph/workflow
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile(checkpointer=memory)







