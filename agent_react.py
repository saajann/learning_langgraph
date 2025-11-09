# reason + act agent

from typing import TypedDict, Sequence, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """add two numbers"""
    return a + b

tools = [add]

model = ChatOllama(model='mistral:latest').bind_tools(tools)

def process_node(state: AgentState) -> AgentState:
    """process node"""
    system_prompt = SystemMessage('You must use the available tools for all math operations. Do not compute results yourself. Use the `add` tool instead')

    response = model.invoke([system_prompt] + state['messages'])

    return {'messages':[response]}

def should_continue(state: AgentState) -> AgentState:
    """check if you need a tool"""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'continue'
    return 'end'

graph = StateGraph(AgentState)

graph.add_node("process", process_node)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge(START, "process")
graph.add_edge("tools", "process")

agent = graph.compile()

def print_stream(stream): 
    for s in stream:
        message = s["messages"][-1]
        if isinstance (message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12")]}

print_stream(agent.stream(inputs, stream_mode="values"))