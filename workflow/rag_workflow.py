import logging
import os
from typing_extensions import TypedDict, NotRequired

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from db_service.faiss_store import search_documents_v2

class State(TypedDict):
    conversation_history: list
    input: str
    messages: list
    output: str
    task_completed: bool
    retrieved_answers: NotRequired[int]  # count of retrieved answers, defaults to 5
    retrieved_docs: NotRequired[list]  # raw retrieved documents with similarity scores

# Global LLM instance for better performance
_agent = None

def get_agent():
    """Get or create global LLM agent instance."""
    global _agent
    if _agent is None:
        _agent = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
    return _agent


async def rag_retrieve_node(state: State) -> State:
    """Retrieve relevant documents using FAISS vector search."""
    query = state["input"]
    k = state.get("retrieved_answers", 5)
    logging.debug(f"number of retrieved answers: {k}")
    # Retrieve documents with similarity scores
    retrieved_docs = search_documents_v2(query, k)
    logging.debug(f"retrieved_docs: {retrieved_docs}")
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[文档{i}] {doc['raw_doc']}")

    context = "\n\n".join(context_parts)

    new_state = state.copy()
    new_state["retrieved_docs"] = retrieved_docs
    new_state["conversation_history"] = context
    new_state["output"] = context  # Pass context to next node
    logging.debug(f"Retrieved {len(retrieved_docs)} documents")
    return new_state


async def rag_generate_node(state: State) -> State:
    """Generate professional summary using LLM with retrieved context."""
    query = state["input"]
    context = state.get("conversation_history", "")

    # Build a professional prompt for summarization
    prompt = f"""You are a professional information summarization assistant. Please provide a professional and accurate summary of the user's question based on the following retrieved document content.

    User question: {query}

    Retrieved relevant documents:
    {context}

    Requirements:
    1. Answer the user's question directly; do not use conversational openings (such as "The question you raised is very interesting," etc.)
    2. Base the summary strictly on the retrieved document content; do not add information not present in the documents
    3. Use professional and objective language
    4. If there are differing viewpoints in the documents, present them objectively
    5. The summary should be well-structured and highlight key points

    Please begin summarizing:"""


    agent = get_agent()
    response = await agent.ainvoke([HumanMessage(content=prompt)])
    logging.debug(f"RAG summary generated")

    state_copy = state.copy()
    state_copy['output'] = response.content
    return state_copy


# Build async workflow
workflow = StateGraph(State)
workflow.add_node("rag_retrieve_node", rag_retrieve_node)
workflow.add_node("rag_generate_node", rag_generate_node)

workflow.add_edge(START, "rag_retrieve_node")
workflow.add_edge("rag_retrieve_node", "rag_generate_node")
workflow.add_edge("rag_generate_node", END)

rag_graph = workflow.compile()