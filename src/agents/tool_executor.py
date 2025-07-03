# src/agents/tool_executor.py
from src.models import State
from src.tools.math_tools import MathToolkit


class ToolExecutor:
    def __init__(self):
        self.toolkit = MathToolkit()

    def run(self, state: State) -> State:
        messages = state["messages"]
        tool_calls = state.get("tool_calls", [])

        if not tool_calls:
            return {"messages": messages, "tool_calls": []}

        last_response = messages[-1]  # LLM cevabı
        tool_outputs = []

        for call in tool_calls:
            tool_name = call["name"]
            args = call.get("args", {})
            a = args.get("a")
            b = args.get("b")

            tool = self.toolkit.get_tool_by_name(tool_name)
            result = tool(a, b)
            tool_outputs.append(f"✅ Tool `{tool_name}` sonucu: {result}")

        # Tool çıktısını yeni bir LLM mesajı gibi sonuna ekle
        updated_messages = messages + tool_outputs

        return {
            "messages": updated_messages,
            "tool_calls": [],  # İşlendi, temizleniyor
        }
