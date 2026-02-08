import logging
from fastapi import APIRouter
# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from db_service.history_conversations import load_history_conversation
from db_service.save_conversations import *
from db_service.db import save_conversation_sql
from chat_langchain import app as langgraph_app
import uuid
import asyncio
from workflow.team_leader_workflow import graph
from workflow.react_workflow import react_graph, run_react
from typing import Dict, Any

app = FastAPI(title="RAG Q&A Backend (dev)")

router = APIRouter()


@router.get("/health")
def read_root():
    return {"message": "Hello, world! Backend is running."}


@router.post("/askLLM")
async def ask_llm(model_name: str = "deepseek-reasoner", question: str = "你好", session_id: str = None):
    # 如果没有 session_id，说明是新会话
    is_new_session = not session_id
    if is_new_session:
        session_id = str(uuid.uuid4())

    # 加载历史对话，返回完整的消息列表
    messages = load_history_conversation(question, session_id)

    # 使用 LangGraph 应用处理请求
    # 构造输入消息
    input_messages = {
        "messages": messages
    }
    logging.debug(f"input_messages: {input_messages}")

    # 配置线程 ID 用于会话记忆
    config = {"configurable": {"thread_id": session_id}}

    # 异步调用 LangGraph 应用
    result = await langgraph_app.ainvoke(input_messages, config=config)

    # 提取 AI 回复
    answer_text = ""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            answer_text = message.content
            break

    # 保存对话历史到数据库 (在后台异步执行)
    save_conversation_json(session_id, question, answer_text, model_name)
    save_conversation_sql(session_id, question, answer_text, model_name)

    # 返回结果包含 session_id
    return JSONResponse(status_code=200, content={
        "session_id": session_id,
        "answer": answer_text
    })


# @router.post("/embedding")
# def embedding_text(text: str = "你好"):
#     return JSONResponse(status_code=200, content={
#         "embedding_result": embedding_processor(text)
#     })


@router.post("/team-leader-task")  # will rename to chat-task
async def team_leader_task(question: str, retrieved_answers:int=5):
    """
    API endpoint that invokes the team leader workflow to handle user tasks
    """
    try:
        # Initialize the state for the workflow
        initial_state = {
            "input": question,
            "output": "",
            "conversation_history": [],
            "messages": [],
            "retrieved_answers": retrieved_answers,
        }
        logging.debug(f"Initial state: {initial_state}")

        # Run the workflow asynchronously
        final_state = await graph.ainvoke(initial_state)

        # Extract tool output and retrieved docs from messages
        output = ""
        retrieved_docs = []
        for msg in final_state.get("messages", []):
            if hasattr(msg, 'content') and msg.content:
                # Parse llm_rag JSON output
                try:
                    import json
                    content = msg.content
                    if isinstance(content, str) and content.startswith('{'):
                        parsed = json.loads(content)
                        if "summary" in parsed:
                            output = parsed.get("summary", content)
                            retrieved_docs = parsed.get("retrieved_docs", [])
                        else:
                            output = content
                    else:
                        output = content
                except (json.JSONDecodeError, TypeError):
                    output = content
                break

        final_answer = output or "未能生成有效回答"
        print(f"Retrieved docs: {retrieved_docs}")

        # Build messages_summary with structured document data
        messages_summary = []
        if retrieved_docs:
            # Format retrieved documents as structured data
            for doc in retrieved_docs:
                messages_summary.append({
                    "content": {
                        "raw_doc": doc.get("raw_doc", ""),
                        "similarity": doc.get("similarity", 0.0)
                    }
                })
        else:
            # If no retrieved docs, include a simple message summary
            for msg in final_state.get("messages", []):
                if hasattr(msg, 'content') and msg.content:
                    messages_summary.append({
                        "content": str(msg.content)[:500]
                    })

        return JSONResponse(
            status_code=200,
            content={
                "question": question,
                "answer": final_answer,
                "retrieved_answers": final_state.get("retrieved_answers"),
                "messages_summary": messages_summary,
            }
        )
    except Exception as e:
        logging.error(f"Error in team_leader_task: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时发生错误: {str(e)}"
            }
        )



@router.post("/react-ask")
async def react_ask(question: str, max_iterations: int = 10, expand_query_num: int=3, retrieved_answers: int = 5, session_id: str = None):
    """
    ReAct Agent API endpoint that uses reasoning-acting loop to handle user queries.

    The ReAct agent will:
    1. THOUGHT: Analyze the user's request
    2. ACTION: Call appropriate tools (llm_chat, llm_query, llm_rag)
    3. OBSERVATION: Review tool results
    4. ITERATE: Continue until satisfied or max_iterations reached

    Args:
        question: User's query or request
        max_iterations: Maximum number of ReAct iterations (default: 10)
        expand_query_num: Number of query that expand based on question (default: 3)
        reretrieved_answers: Number of documents to retrieve when using RAG (default: 5)
        session_id: Optional session ID for conversation tracking

    Returns:
        JSON response with:  
            - question: Original question
            - answer: Final answer from the agent
            - iteration_count: Number of iterations performed
            - tool_calls_summary: Summary of tools used
            - messages_count: Total messages in conversation
            - session_id: Session identifier
    """
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logging.info(f"ReAct request - Session: {session_id}, Question: {question[:50]}...")

        # Run the ReAct workflow
        result = await run_react(question, max_iterations=max_iterations, expand_query_num=expand_query_num, retrieved_answers=retrieved_answers, session_id=session_id)

        # Check if successful
        if not result.get("success"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": result.get("error", "Unknown error"),
                    "question": question,
                    "session_id": session_id
                }
            )

        # Extract messages and find the final answer
        messages = result.get("messages", [])
        answer = ""

        # Find the last AI message as the final answer
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    # This is the final answer (no tool calls)
                    answer = msg.content
                break

        # If no answer found, try to get from result output
        if not answer:
            answer = result.get("output", "")

        # Build messages_summary with retrieved documents (same format as team-leader-task)
        retrieved_docs = result.get("retrieved_docs", [])
        messages_summary = []

        if retrieved_docs:
            # Format retrieved documents as structured data
            for doc in retrieved_docs:
                messages_summary.append({
                    "content": {
                        "raw_doc": doc.get("raw_doc", ""),
                        "similarity": doc.get("similarity", 0.0)
                    }
                })
        else:
            # If no retrieved docs, include conversation summary
            for msg in messages:
                if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    messages_summary.append({
                        "content": str(msg.content)[:500]
                    })
                    break

        # Get retrieved_answers count and iteration_count
        retrieved_count = result.get("retrieved_answers", retrieved_answers)
        iteration_count = result.get("iteration_count", 0)

        # Save conversation to database
        save_conversation_json(session_id, question, answer, "react-agent")
        save_conversation_sql(session_id, question, answer, "react-agent")

        logging.info(f"ReAct completed - Session: {session_id}, Iterations: {iteration_count}")

        return JSONResponse(
            status_code=200,
            content={
                "question": question,
                "answer": answer,
                "retrieved_answers": retrieved_count,
                "messages_summary": messages_summary,
                "iteration_count": iteration_count,
                "session_id": session_id
            }
        )

    except Exception as e:
        logging.error(f"Error in react_ask: {str(e)}")
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时发生错误: {str(e)}",
                "question": question,
                "session_id": session_id
            }
        )


if __name__ == "__main__":
    pass