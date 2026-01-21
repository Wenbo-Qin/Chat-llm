import logging
from fastapi import APIRouter
# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from embedding import embedding
from service.history_conversations import load_history_conversation
from service.save_conversations import *
from service.db import save_conversation_sql
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
def ask_llm(model_name: str = "deepseek-reasoner", question: str = "你好", session_id: str = None):
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

    # 调用 LangGraph 应用
    result = langgraph_app.invoke(input_messages, config=config)

    # 提取 AI 回复
    answer_text = ""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            answer_text = message.content
            break

    # 保存对话历史到数据库
    save_conversation_json(session_id, question, answer_text, model_name)
    save_conversation_sql(session_id, question, answer_text, model_name)

    # 返回结果包含 session_id
    return JSONResponse(status_code=200, content={
        "session_id": session_id,
        "answer": answer_text
    })


@router.post("/embedding")
def embedding_text(text: str = "你好"):
    return JSONResponse(status_code=200, content={
        "embedding_result": embedding(text)
    })


@router.post("/mcp")
def mcp():
    return None


@router.post("/team-leader-task")  # will rename to chat-task
async def team_leader_task(question: str):
    """
    API endpoint that invokes the team leader workflow to handle user tasks
    """
    try:
        # Initialize the state for the workflow
        initial_state = {
            # "current_step": "start",
            "input": question,
            "output": "",
            "conversation_history": [],
            "task_completed": False,
            "messages": [],  # Required for LangGraph tools_condition
            # "iteration_count": 0
        }

        # Run the workflow asynchronously
        final_state = await graph.ainvoke(initial_state, config={"max_iterations": 2})

        # Extract the final answer from the workflow result
        # Try to get the content from the last tool message in messages
        messages = final_state.get("messages", [])
        final_answer = "未能生成有效回答"
        
        # Look for the last AIMessage with content in the messages
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                try:
                    # Check if content is a JSON string that needs parsing
                    import json
                    content = msg.content
                    if isinstance(content, str) and content.startswith('{'):
                        # Parse the JSON string and extract the output
                        parsed_content = json.loads(content)
                        final_answer = parsed_content.get('output', content)
                    else:
                        # Content is already a string
                        final_answer = content
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the content as is
                    final_answer = msg.content
                break
        
        # Extract conversation history for context
        conversation_history = final_state.get("conversation_history", [])
        
        return JSONResponse(
            status_code=200,
            content={
                "question": question,
                "answer": final_answer,
                "conversation_history": conversation_history,
                "task_completed": final_state.get("task_completed", False)
            }
        )
    except Exception as e:
        logging.error(f"Error in team_leader_task: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时发生错误: {str(e)}"
            }
        )


if __name__ == "__main__":
    pass