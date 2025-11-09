from typing import List, TypedDict, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# state

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# nodes 

llm = ChatOllama(model='gemma3:1b')

def process_node(state: AgentState) -> AgentState:
    """node to conversate with an LLM"""

    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(response.content))

    print('* Answer: ' + response.content)

    return state

# graph 

graph = StateGraph(AgentState)

graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# memory / run agent

conversation_history = []
user_input = input('* Question: ')

while user_input != 'bye':
    conversation_history.append(HumanMessage(user_input))
    response = agent.invoke({'messages': conversation_history})
    conversation_history = response['messages']

    user_input = input('* Question: ')