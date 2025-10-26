from fastapi import APIRouter
# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from service.history_conversations import load_history_conversation
from service.save_conversations import *
from service.db import save_conversation_sql
from chat_langchain import app as langgraph_app
import uuid

app = FastAPI(title="RAG Q&A Backend (dev)")

router = APIRouter()


@router.get("/health")
def read_root():
    return {"message": "Hello, world! Backend is running."}


@router.get("/askLLM")
def ask_llm(model_name: str = "deepseek-reasoner", question: str = "你好", session_id: str = None):
    # 如果没有 session_id，说明是新会话
    is_new_session = not session_id
    if is_new_session:
        session_id = str(uuid.uuid4())
    question = load_history_conversation(question, session_id)
    # 使用 LangGraph 应用处理请求
    # 构造输入消息
    input_messages = {
        "messages": [HumanMessage(content=question)]
    }
    print("input_messages", input_messages)

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
