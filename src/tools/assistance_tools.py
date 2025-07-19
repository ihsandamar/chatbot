
from typing import Dict
from langchain_core.tools import tool
from langchain_core.tools.base import BaseToolkit
from src.models import State
from langgraph.types import Command, interrupt
from langchain_core.tools import BaseTool

@tool
def human_assistance_tool(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    result = human_response.get("result")
    print(result)
    return human_response.get("query", "No response received from human assistance.")


@tool
def update_state_tool(state: State, key: str, value: str) -> State:
    """Update the state with a new key-value pair."""
    state[key] = value
    return state



class AssistanceToolkit(BaseToolkit):

    def __init__(self):
        """Initialize the AssistanceToolkit with available tools."""
        super().__init__(name="AssistanceToolkit", description="Toolkit for human assistance operations.")
    
    def get_tools(self) -> list[BaseTool]:
        """Get the tools available in this toolkit."""
        return [human_assistance_tool, update_state_tool]
