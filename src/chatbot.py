from langchain_openai import ChatOpenAI
from IPython.display import display, Image
from langgraph.graph import StateGraph, START, END
from os import environ
from dotenv import load_dotenv
from src.models import State


# Initialize the graph builder 
graph_builder = StateGraph(State)


load_dotenv()
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")

# Add a Chatbot Node
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Create the graph/workflow
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

