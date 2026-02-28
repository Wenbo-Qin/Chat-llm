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
import dashscope
from http import HTTPStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(TypedDict):
    conversation_history: list
    input: str
    messages: list
    output: str
    task_completed: bool
    expanded_queries: NotRequired[list]  # List of expanded queries
    expand_query_num: NotRequired[int]  # Number of query that expand based on question
    retrieve_k: NotRequired[int]  # Initial number of documents to retrieve per query
    retrieved_docs: NotRequired[list]  # raw retrieved documents with similarity scores
    reranked_docs: NotRequired[list]  # reranked documents after rerank node
    enable_rerank: NotRequired[bool]  # Whether to enable rerank, defaults to True
    rerank_top_n: NotRequired[int]  # Number of documents to return after rerank

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
    5. The number of expanded queries should be {state.get("expand_query_num")}.

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
    k = state.get("retrieve_k", 5)
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

    new_state = state.copy()
    new_state["expanded_queries"] = expanded_queries
    new_state["retrieved_docs"] = unique_docs
    logging.debug(f"Retrieved {len(unique_docs)} unique documents")
    return new_state


async def rag_rerank_node(state: State) -> State:
    """Rerank retrieved documents using qwen3-rerank."""
    retrieved_docs = state.get("retrieved_docs", [])
    original_query = state["input"].split("\n")[0]  # Get original query without conversation history

    # Check if rerank is enabled, defaults to True
    enable_rerank = state.get("enable_rerank", True)

    # Use rerank_top_n if provided, otherwise fall back to retrieve_k
    top_n = state.get("rerank_top_n") or state.get("retrieve_k", 5)

    if not retrieved_docs:
        new_state = state.copy()
        new_state["reranked_docs"] = []
        new_state["conversation_history"] = ""
        return new_state

    # Validate and adjust top_n if it exceeds available documents
    actual_doc_count = len(retrieved_docs)
    if top_n > actual_doc_count:
        logging.warning(f"rerank_top_n ({top_n}) exceeds actual retrieved documents ({actual_doc_count}), adjusting to {actual_doc_count}")
        top_n = actual_doc_count

    # If rerank is disabled, use original docs directly
    if not enable_rerank:
        logging.debug(f"Rerank disabled, using original {min(len(retrieved_docs), top_n)} documents")
        selected_docs = retrieved_docs[:top_n]
        reranked_docs = [
            {"raw_doc": doc["raw_doc"], "rerank_score": doc.get("similarity", 0)}
            for doc in selected_docs
        ]
    else:
        logging.debug(f"Reranking {len(retrieved_docs)} documents to top {top_n}")

        # Prepare documents for rerank API
        documents = [doc["raw_doc"] for doc in retrieved_docs]

        try:
            # Call qwen3-rerank API
            resp = dashscope.TextReRank.call(
                model="qwen3-rerank",
                query=original_query,
                api_key=os.getenv("QWEN_API_KEY"),
                documents=documents,
                top_n=top_n,
                return_documents=True,
                instruct="Given a web search query, retrieve relevant passages that answer the query."
            )

            if resp.status_code == HTTPStatus.OK:
                # Build reranked docs with scores
                reranked_docs = []
                for result in resp.output.results:
                    doc_index = result.index
                    relevance_score = result.relevance_score
                    reranked_docs.append({
                        "raw_doc": documents[doc_index],
                        "rerank_score": relevance_score,
                        "doc_id": retrieved_docs[doc_index].get("doc_id", doc_index)
                    })
                logging.debug(f"Reranked to {len(reranked_docs)} documents")
            else:
                logging.warning(f"Rerank API failed: {resp.message}, using original order")
                # Fallback to original docs
                reranked_docs = [
                    {"raw_doc": doc["raw_doc"], "rerank_score": doc.get("similarity", 0)}
                    for doc in retrieved_docs[:top_n]
                ]
        except Exception as e:
            logging.error(f"Rerank error: {e}, using original order")
            # Fallback to original docs
            reranked_docs = [
                {"raw_doc": doc["raw_doc"], "rerank_score": doc.get("similarity", 0)}
                for doc in retrieved_docs[:top_n]
            ]

    # Build context from reranked documents
    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        context_parts.append(f"[文档{i}] {doc['raw_doc']}")

    context = "\n\n".join(context_parts)

    new_state = state.copy()
    new_state["reranked_docs"] = reranked_docs
    new_state["conversation_history"] = context
    new_state["output"] = context  # Pass context to next node
    return new_state


async def rag_generate_node(state: State) -> State:
    """Generate professional summary using LLM with retrieved context."""
    query = state["input"]
    retrieved_docs = state["retrieved_docs"]
    expanded_queries = state.get("expanded_queries")
    logger.info(f"Number of generated expanded queries: {len(expanded_queries)}")
    logger.info(f"Generating expanded queries: {expanded_queries}")
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
workflow.add_node("rag_rerank_node", rag_rerank_node)
workflow.add_node("rag_generate_node", rag_generate_node)

workflow.add_edge(START, "rag_query_expand_node")
workflow.add_edge("rag_query_expand_node", "rag_retrieve_node")
workflow.add_edge("rag_retrieve_node", "rag_rerank_node")
workflow.add_edge("rag_rerank_node", "rag_generate_node")
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