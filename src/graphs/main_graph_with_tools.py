# src/graphs/main_graph_with_tools.py
from langgraph.graph import StateGraph, START, END
from src.models.models import LLM, State
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
from src.services.app_logger import log


@register_graph("main_graph_with_tools")
class MainGraph(BaseGraph):
    """Main graph that orchestrates tool‑aware LLM interactions.

    Added rich/structlog‑backed logging at each critical step so that
    execution flow is instantly visible in your console (or file when
    `LOG_TO_FILE=1`).
    """

    def __init__(self, llm: LLM):
        # Keep a class‑level logger bound with static context
        self._logger = log.get(module="graphs", file="main_graph_with_tools", cls="MainGraph")
        self._logger.debug("Initializing MainGraph instance")
        super().__init__(llm=llm, state_class=State)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_graph(self):
        """Builds and compiles the LangGraph workflow with rich logging."""
        lg = self._logger.bind(method="build_graph")
        lg.debug("Starting graph build")

        # --- Checkpointing -------------------------------------------------
        memory = MemorySaver()
        lg.debug("MemorySaver initialized")

        # --- Tools ---------------------------------------------------------
        tools = []
        math_tools = MathToolkit().get_tools()
        tools += math_tools
        lg.debug("MathToolkit tools loaded", count=len(math_tools))

        date_tools = DateToolkit().get_tools()
        tools += date_tools
        lg.debug("DateToolkit tools loaded", count=len(date_tools))

        assistance_tools = AssistanceToolkit().get_tools()
        tools += assistance_tools
        lg.debug("AssistanceToolkit tools loaded", count=len(assistance_tools))
        lg.debug("Total tools", total=len(tools))

        # --- Agent ---------------------------------------------------------
        agent = create_react_agent(
            model=self.llm.get_chat(),
            tools=tools,
        )
        lg.debug("React agent created")

        # --- Helper: final assistant step ----------------------------------
        def assistant_final(state: State) -> State:
            alog = lg.bind(fn="assistant_final")
            alog.debug("assistant_final invoked", messages=len(state["messages"]))
            response = self.llm.get_chat().invoke(state["messages"])
            alog.info("assistant_final response generated")
            return {"messages": state["messages"] + [response]}

        # --- Tool execution node -------------------------------------------
        tool_node = ToolNode(tools)
        lg.debug("ToolNode created")

        # --- Graph definition ----------------------------------------------
        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tool_executor", tool_node)
        graph.add_node("assistant_final", assistant_final)
        lg.debug("Graph nodes added")

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            lambda s: "tool_executor" if s["messages"][-1].additional_kwargs.get("function_call") else "assistant_final",
            path_map={
                "tool_executor": "tool_executor",
                "assistant_final": "assistant_final",
            },
        )
        lg.debug("Conditional edges wired")

        graph.add_edge("tool_executor", "assistant_final")
        graph.add_edge("assistant_final", END)
        lg.debug("Static edges wired")

        compiled_graph = graph.compile(name="main_graph_with_tools", checkpointer=memory)
        lg.info("Graph compiled successfully")
        return compiled_graph
