from typing import TypedDict, List, Any
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---- TOOL ---------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """İki sayıyı çarpar."""
    return a * b

tools = [multiply]

# ---- STATE --------------------------------------------------
class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]

# ---- LLM ----------------------------------------------------
from src.services.config_loader import ConfigLoader
config = ConfigLoader.load_config()

llm = ChatOpenAI(
    model=config.llm.model,
    temperature=config.llm.temperature,
    api_key=config.llm.api_key
)

# (create_react_agent de kullanabilirsiniz ama en kısa yol:)
assistant_decide = create_react_agent(model=llm, tools=tools)

# ---- TOOL ÇALIŞTIRICI --------------------------------------
tool_exec = ToolNode(tools)

# ---- FİNAL CEVAP DÜĞÜMÜ -------------------------------------
def assistant_final(state: ChatState) -> ChatState:
    # Son mesaj listesine bakarak tek seferlik cevap üretiyoruz
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# ---- GRAF ---------------------------------------------------
def make_graph():
    memory = MemorySaver()
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("assistant_decide", assistant_decide)
    graph_builder.add_node("tool_exec", tool_exec)
    graph_builder.add_node("assistant_final", assistant_final)

    graph_builder.add_edge(START, "assistant_decide")

    graph_builder.add_conditional_edges(
        "assistant_decide",
        lambda s: "tool" if s["messages"][-1].additional_kwargs.get("function_call") else "final",
        path_map={                       # 🔹 DEĞİŞEN PARAMETRE ADI
            "tool": "tool_exec",
            "final": "assistant_final",
        },
    )

    graph_builder.add_edge("tool_exec", "assistant_final")
    graph_builder.add_edge("assistant_final", END)

    graph = graph_builder.compile(name="make_graph", checkpointer=memory)
    return graph

# ---- TEST ---------------------------------------------------
initial_state = {
    "messages": [HumanMessage(content="12 ile 7'yi çarpar mısın?")]
}
configurable = {"configurable": {"thread_id": uuid.uuid4().hex, "graph_name": "tool_calling_graph"}}
result_state = make_graph().invoke(initial_state, config=configurable)
print(result_state["messages"][-1].content)  # ➜  “Tabii! 12 × 7 = 84.”




