# src/nodes/decision_nodes.py
from typing import Literal
from langgraph.graph import END
from src.models.models import State

class DecisionNodes:
    """Karar verme nodeları"""
    
    @staticmethod
    def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]: # type: ignore
        """Devam etmeli mi kontrol et"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # SubmitFinalAnswer varsa bitir
        if getattr(last_message, "tool_calls", None):
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "SubmitFinalAnswer":
                    return END
        
        # Hata varsa sorgu üretimine geri dön
        if last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"