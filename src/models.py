from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Define the state schema 
class State(TypedDict):
    messages: Annotated[list, add_messages]




class LLM:
    providder = "openai"
    model = "gpt-3.5-turbo"
    temperature = 0.0
    api_key = None

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0, api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key


    def get_chat(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
