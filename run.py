from src.chatbot import graph
from src.graphs.main_graph import MainGraph
from src.models import LLM
from src.config import OPENAI_API_KEY


llm = LLM(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
graph = MainGraph(llm=llm).build_graph()

if __name__ == "__main__":
    result = graph.invoke({"messages": ["Hello, how are you?"]})
    print("Chatbot Response:", result)
