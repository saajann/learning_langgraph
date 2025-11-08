from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    questions: List[HumanMessage]
    answers: str
    
llm = ChatOllama(model="mistral:latest")

def process_node(state: AgentState) -> AgentState:
    """node to ask a question to an LLM"""

    state['answers'] = llm.invoke(state['questions']).content
    print('* answer: ' + state['answers'])

    return state

graph = StateGraph(AgentState)

graph.add_node("process", process_node)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input('* question: ')
while user_input != 'bye':
    agent.invoke({"questions": [HumanMessage(content=user_input)]})
    user_input = input('* question: ')