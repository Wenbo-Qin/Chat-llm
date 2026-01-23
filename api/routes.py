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
            "task_completed": False,
            "messages": [],
            "retrieved_answers": retrieved_answers,
        }
        logging.debug(f"Initial state: {initial_state}")

        # Run the workflow asynchronously
        final_state = await graph.ainvoke(initial_state)

        # Extract the final answer from the workflow result
        final_answer = final_state.get("output", "未能生成有效回答")

        # Extract retrieved documents if available
        retrieved_docs = final_state.get("retrieved_docs", [])
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
                "task_completed": final_state.get("task_completed"),
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



if __name__ == "__main__":
    pass