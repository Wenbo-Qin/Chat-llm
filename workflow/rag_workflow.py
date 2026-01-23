import logging
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, NotRequired
from db_service.faiss_store import search_documents_v2
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import os

class State(TypedDict):
    conversation_history: list
    input: str
    messages: list
    output: str
    task_completed: bool
    conversation_history: list  # store all conversation history
    retrieved_answers: NotRequired[int]  # count of retrieved answers, defaults to 5

# def embedding_text(text: str = "你好") -> str:
#     result = embedding_processor(text)
#     return result
def rag_retrieve_node(state: State) -> State:
    query = state["input"]
    k = state.get("retrieved_answers", 5)

    content = search_documents_v2(query, k)

    new_state = state.copy()
    new_state["output"] = content
    new_state["conversation_history"] = content
    logging.debug(new_state)
    return new_state
def rag_generate_node(state: State) -> State:
    # Ensure retrieved_answers has a default value if not present
    retrieved_answers = state.get('retrieved_answers')
    
    logging.debug(state)
    agent = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), 
                        base_url="https://api.deepseek.com",
                        model = "deepseek-chat")    
    response = agent.invoke([HumanMessage(content=state['output'])])
    logging.debug(f"agent response: {response}")
    state_copy = state.copy()
    state_copy['conversation_history'] = state.get('conversation_history')
    state_copy['output'] = response.content
    return state_copy

workflow = StateGraph(State)
workflow.add_node("rag_retrieve_node", rag_retrieve_node)
workflow.add_node("rag_generate_node", rag_generate_node)

workflow.add_edge(START, "rag_retrieve_node")
workflow.add_edge("rag_retrieve_node", "rag_generate_node")
workflow.add_edge("rag_generate_node", END)

rag_graph = workflow.compile()