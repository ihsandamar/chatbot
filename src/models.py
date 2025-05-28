from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, AnyMessage

# Define the state schema 
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]




class LLM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key


    def get_chat(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
    
    
    def send(self, messages: State) -> str:
        chat = self.get_chat()
        response = chat.invoke(messages)
        return response.content if hasattr(response, "content") else response.get("content", "")
