# src/agents/llm_agent.py
from src.models.models import State
from src.tools.math_tools import MathToolkit


class LLMAgent:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools

    def run(self, state: State) -> State:
        response = self.llm_with_tools.invoke(state["messages"])
        # Tool çağrısı varsa tool_calls'a ekle
        return {
            "messages": state["messages"] + [response],
            "tool_calls": getattr(response, "tool_calls", []),
        }
