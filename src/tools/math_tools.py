from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import BaseToolkit

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

class MathToolkit(BaseToolkit):
    """Toolkit for mathematical operations."""
    def __init__(self):
        """Initialize the MathToolkit with available tools."""
        super().__init__(name = "MathToolkit", description="Toolkit for basic math operations like addition, multiplication, and division.")

    def get_tools(self):
        return [add, multiply, divide]
    
    def get_tool_by_name(self, name: str) -> callable:
        """Get a tool by its name.
        Returns the tool function if found, otherwise raises a ValueError.
        """
        tools = self.get_tools()
        for tool in tools:
            if tool.__name__ == name:
                return tool
        raise ValueError(f"Tool with name {name} not found.")

