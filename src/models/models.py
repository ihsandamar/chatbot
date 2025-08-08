from dataclasses import dataclass
from typing import NotRequired, Optional, TypedDict, Annotated, List, Literal, Any
from langgraph.graph.message import add_messages, AnyMessage

# Define the state schema 
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Text2SQL specific fields
    user_query: NotRequired[str]  # User's question
    all_tables: NotRequired[List[str]]  # All database tables
    relevant_tables: NotRequired[List[Any]]  # Tables relevant to the user query with schema
    table_schemas: NotRequired[str]  # Raw schema information
    query_parameters: NotRequired[Any]  # Extracted SQL query parameters
    generated_sql: NotRequired[str]  # Generated SQL query
    validated_sql: NotRequired[str]  # Validated SQL query
    is_valid: NotRequired[bool]  # Whether SQL passed validation
    sql_result: NotRequired[str]  # SQL execution result
    is_error: NotRequired[bool]  # Whether there was an execution error
    error_message: NotRequired[str]  # Error message if execution failed
    fixed_sql: NotRequired[str]  # Fixed SQL after error correction



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



