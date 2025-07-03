# src/graphs/main_graph_with_tools.py
from langgraph.graph import StateGraph, START, END
from src.models import LLM, State
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.agents.llm_agent import LLMAgent
from src.agents.tool_executor import ToolExecutor
from src.tools.math_tools import MathToolkit
from langgraph.checkpoint.memory import MemorySaver


@register_graph("main_graph_with_tools")
class MainGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=State)

    def build_graph(self):
        memory = MemorySaver()

        # LLM + Tool binding
        llm_with_tools = self.llm.get_chat().bind_tools(MathToolkit().get_tools())

        # Agent ve Executor tanımları
        llm_agent = LLMAgent(llm_with_tools)
        tool_executor = ToolExecutor()

        graph = StateGraph(State)
        graph.add_node("llm_agent", llm_agent.run)
        graph.add_node("tool_executor", tool_executor.run)

        graph.set_entry_point("llm_agent")

        # Eğer tool call varsa executor'a yönlendir
        def route(state: State) -> str:
            if state.get("tool_calls"):
                return "tool_executor"
            return END

        graph.add_conditional_edges("llm_agent", route)
        graph.add_edge("tool_executor", END)

        return graph.compile(name="main_graph_with_tools", checkpointer=memory)
