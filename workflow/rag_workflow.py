import logging
import os
from typing_extensions import TypedDict, NotRequired

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
# This allows the file to be run directly from the workflow directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    expanded_queries: NotRequired[list]  # List of expanded queries
    expand_query_num: NotRequired[int]  # Number of query that expand based on question
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

async def rag_query_expand_node(state: State) -> State:
    """Expand query to include relevant context."""
    
    prompt = f"""You are a helpful assistant that expands a user's question to include relevant context
    {state["input"]}.
    The number of expanded queries should be {state.get("expand_query_num")}.

    Requirements:
    1. DO NOT repeat the original question in each expanded query
    2. Each query should explore a DIFFERENT aspect (definition, types, applications, principles, techniques, etc.)
    3. Be concise and focused on the user's question
    4. Output each query on a separate line without numbering

    Please expand the user's question to include relevant context.
    Please begin expanding the question:"""
    
    agent = get_agent()
    response = await agent.ainvoke([HumanMessage(content=prompt)])
    expand_query = response.content
    # print(f"Expanded queries:\n{expand_query}", sep="\n")
    query = state["input"]
    context = state.get("conversation_history", "")
    new_state = state.copy()
    new_state["expanded_queries"] = expand_query
    new_state["input"] = f"{query}\n{context}"
    return new_state
async def rag_retrieve_node(state: State) -> State:
    """Retrieve relevant documents using FAISS vector search for multiple queries."""
    k = state.get("retrieved_answers", 5)
    logging.debug(f"number of retrieved answers per query: {k}")

    # Get original query and expanded queries
    original_query = state["input"]
    expanded_queries_str = state.get("expanded_queries", "")

    # Parse expanded queries string into list
    expanded_queries = [q.strip() for q in expanded_queries_str.strip().split('\n') if q.strip()]
    # Combine all queries
    all_queries = [original_query] + expanded_queries
    logging.debug(f"Retrieving for {len(all_queries)} queries")

    # Retrieve k documents for each query
    all_retrieved_docs = []
    for i, query in enumerate(all_queries, 1):
        docs = search_documents_v2(query, k)
        all_retrieved_docs.extend(docs)
        logging.debug(f"Query {i}/{len(all_queries)} retrieved {len(docs)} documents")

    # Deduplicate by doc_id if available, otherwise by raw_doc
    seen_doc_ids = set()
    unique_docs = []
    for doc in all_retrieved_docs:
        doc_id = doc.get("doc_id", doc.get("raw_doc"))
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            unique_docs.append(doc)

    logging.debug(f"Total retrieved: {len(all_retrieved_docs)}, After deduplication: {len(unique_docs)}")

    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(unique_docs, 1):
        context_parts.append(f"[文档{i}] {doc['raw_doc']}")

    context = "\n\n".join(context_parts)

    new_state = state.copy()
    new_state["expanded_queries"] = expanded_queries
    new_state["retrieved_docs"] = unique_docs
    new_state["conversation_history"] = context
    new_state["output"] = context  # Pass context to next node
    logging.debug(f"Retrieved {len(unique_docs)} unique documents")
    return new_state


async def rag_generate_node(state: State) -> State:
    """Generate professional summary using LLM with retrieved context."""
    query = state["input"]
    retrieved_docs = state["retrieved_docs"]
    expanded_queries = state.get("expanded_queries")
    # Build a professional prompt for summarization
    prompt = f"""You are a professional information summarization assistant. Please provide a professional and accurate summary of the user's question and expanded queried based on the following retrieved document content.

    User question: {query}
    Expanded queries: {expanded_queries}
    Retrieved relevant documents:
    {retrieved_docs}

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
workflow.add_node("rag_query_expand_node", rag_query_expand_node)  # Placeholder for query expansion
workflow.add_node("rag_retrieve_node", rag_retrieve_node)
workflow.add_node("rag_generate_node", rag_generate_node)

workflow.add_edge(START, "rag_query_expand_node")
workflow.add_edge("rag_query_expand_node", "rag_retrieve_node")
workflow.add_edge("rag_retrieve_node", "rag_generate_node")
workflow.add_edge("rag_generate_node", END)

rag_graph = workflow.compile()

if __name__ == "__main__":
    import asyncio

    async def test_rag():
        state = {
            "input": "什么是机器学习？",
            "expand_query_num": 5,
        }
        result = await rag_graph.ainvoke(state)
        # print(result)

    asyncio.run(test_rag())