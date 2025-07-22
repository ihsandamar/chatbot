# src/graphs/main_graph_with_tools.py
from langgraph.graph import StateGraph, START, END
from src.models import LLM, State
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.agents.llm_agent import LLMAgent
from src.agents.tool_executor import ToolExecutor
from src.tools.assistance_tools import AssistanceToolkit
from src.tools.math_tools import MathToolkit
from src.tools.date_tool import DateToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode


@register_graph("main_graph_with_tools")
class MainGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=State)


    def build_graph(self):
        memory = MemorySaver()

        tools = MathToolkit().get_tools()
        tools += DateToolkit().get_tools()  # Eğer DateToolkit eklemek isterseniz
        tools += AssistanceToolkit().get_tools()  # Yardımcı araçlar

        agent = create_react_agent(
            model=self.llm.get_chat(),
            tools=tools,
        )

        def assistant_final(state: State) -> State:
            # Son mesaj listesine bakarak tek seferlik cevap üretiyoruz

            response = self.llm.get_chat().invoke(state["messages"])
            return {"messages": state["messages"] + [response]}


        tool_node = ToolNode(tools)


        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tool_executor", tool_node)
        graph.add_node("assistant_final", assistant_final)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            lambda s: "tool_executor" if s["messages"][-1].additional_kwargs.get("function_call") else "assistant_final",
            path_map={
                "tool_executor": "tool_executor",
                "assistant_final": "assistant_final",
            },
        )
        graph.add_edge("tool_executor", "assistant_final")
        graph.add_edge("assistant_final", END)

        return graph.compile(name="main_graph_with_tools", checkpointer=memory)
    

