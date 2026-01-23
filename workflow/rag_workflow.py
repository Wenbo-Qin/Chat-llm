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
    retrieved_answers: NotRequired[int]  # count of retrieved answers, defaults to 5

# def embedding_text(text: str = "你好") -> str:
#     result = embedding_processor(text)
#     return result
def agent_generate_response(state: State) -> State:
    # Ensure retrieved_answers has a default value if not present
    retrieved_answers = state.get('retrieved_answers', 5)
    
    logging.debug(state)
    agent = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), 
                        base_url="https://api.deepseek.com",
                        model = "deepseek-chat")    
    response = agent.invoke([HumanMessage(content=state['output'])])
    logging.debug(f"agent response: {response}")
    state_copy = state.copy()
    state_copy['output'] = response.content
    state_copy['retrieved_answers'] = retrieved_answers  # preserve the value
    
    return state_copy

workflow = StateGraph(State)
workflow.add_node("rag_query", search_documents_v2)
workflow.add_node("agent_generate_response", agent_generate_response)

workflow.add_edge(START, "rag_query")
workflow.add_edge("rag_query", "agent_generate_response")
workflow.add_edge("agent_generate_response", END)

rag_graph = workflow.compile()