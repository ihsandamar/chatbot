from os import name
from urllib import response
from src.tools.math_tools import MathToolkit
from langgraph.prebuilt import tools_condition

from langgraph.graph import StateGraph, START, END
from src.graphs.base_graph import BaseGraph
from src.graphs.registry import register_graph
from src.models import LLM, State
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
# from langchain.tools import BaseTool
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import _get_state_args

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    llm = LLM(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.get_chat().bind_tools(MathToolkit().get_tools(), parallel_tool_calls=False)
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

@register_graph("main_graph_with_tools")
class MainGraph(BaseGraph):
    def __init__(self, llm: LLM):
        super().__init__(llm=llm, state_class=State)

    def chatbot(self, state: State):
        return {"messages": [self.llm.get_chat().invoke(state["messages"])]}

    def get_tool_by_name(self, name: str) -> BaseTool:
        """
        Get a tool by its name.
        This is used to find the tool in the tools_by_name dictionary.
        """
        tools = MathToolkit().get_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool with name {name} not found.")

    def assistantbot_with_tools(self, state: State):
        """
        Chatbot function that uses the LLM with tools.
        This function is used to handle messages and tool calls.
        """
        response = self.model_with_tools.invoke(state["messages"])
        print(response.tool_calls)


        # run the tool calls if any
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Execute the tool call and get the result
                tool_call_name = tool_call["name"]
                tool = MathToolkit().get_tool_by_name(tool_call_name)
                tool_args = tool_call["args"]
                tool_result = tool(tool_args["a"], tool_args["b"])


                forcing_tool_call = self.model_with_tools.invoke(
                    state["messages"] + [tool_call],
                    tool_calls=[tool_call]
                )

                return {"messages": state["messages"] + [str(tool_result) + f" (from tool: {tool_call_name})"]}


        updated_messages = state["messages"] + [response]
        return {"messages": updated_messages}

    def build_graph(self):
        """
        Function to build the graph.
        This is useful for testing purposes.
        """
        memory = MemorySaver()
        graph_builder = StateGraph(State)

        self.model_with_tools = self.llm.get_chat().bind_tools(
            MathToolkit().get_tools(),
            # parallel_tool_calls=False,
            # condition=tools_condition("messages", "tool_calls")
        )


        # Create the graph/workflow
        graph_builder.add_node("chatbot", self.assistantbot_with_tools)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile(name = "main_graph_with_tools", checkpointer=memory)

