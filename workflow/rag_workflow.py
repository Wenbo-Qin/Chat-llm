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
    prompt = f"""你是一个专业的信息总结助手。请基于以下检索到的文档内容，对用户的问题进行专业、准确的总结。

用户问题：{query}

检索到的相关文档：
{context}

要求：
1. 直接回答用户的问题，不要使用对话式的开头（如"你提到的问题很有意思"等）
2. 基于检索到的文档内容进行总结，不要添加文档中没有的信息
3. 使用专业、客观的语言
4. 如果文档中有不同的观点，请客观呈现
5. 总结要条理清晰，重点突出

请开始总结："""

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