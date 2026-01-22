from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    conversation_history: list
    input: str
    messages: list
    output: str
    task_completed: bool

def embedding_text(text: str = "你好") -> str:
    return "embedding_result"

workflow = StateGraph(State)
workflow.add_node("rag", embedding_text)

workflow.add_edge(START, "rag", embedding_text)