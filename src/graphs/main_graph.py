from langgraph.graph import StateGraph, START, END
from src.models import LLM, State



class MainGraph:
    """
    Main Graph for the chatbot workflow.
    This graph defines the structure and flow of the chatbot interactions.
    """
    def __init__(self, llm = LLM):
        self.llm = llm.get_chat()

    def chatbot(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}


    def build_graph(self):
        """
        Function to build the graph.
        This is useful for testing purposes.
        """
        graph_builder = StateGraph(State)

        # Create the graph/workflow
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile()







