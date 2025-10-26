from fastapi import APIRouter
# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from service.history_conversations import load_history_conversation
from service.save_conversations import *
from db import save_conversation_sql, query_messages_by_session_id_with_time_order
from chat_langchain import app as langgraph_app
import uuid

app = FastAPI(title="RAG Q&A Backend (dev)")

router = APIRouter()

# # 请求模型：用户发送的问题
# class AskRequest(BaseModel):
#     question: str = Field(..., min_length=1, description="用户的问题，不能为空")
#     session_id: Optional[str] = Field(None, description="可选，会话ID；如果不传服务端会生成")
#     save: bool = Field(True, description="是否保存该问答到数据库（示例字段）")
#
#     # 简单校验示例：去掉首尾空格并确保不是只包含空白
#     @field_validator("question")
#     def strip_and_not_blank(cls, v: str):
#         v2 = v.strip()
#         if not v2:
#             raise ValueError("question 不能全是空白")
#         return v2
#
#
# # 响应模型：服务端返回给前端的结构
# class AskResponse(BaseModel):
#     session_id: str
#     answer: str


@router.get("/health")
def read_root():
    return {"message": "Hello, world! Backend is running."}


# @app.post("/ask", response_model=AskResponse)
# def ask(req: AskRequest):
#     # 如果客户端没传 session_id，则生成一个
#     session = req.session_id or str(uuid.uuid4())
#     # 这里是示例回答逻辑（后面会替换为 RAG+模型返回）
#     answer = f"已收到：{req.question}"
#
#     # 如果业务要求，可以在此做进一步校验并抛出 HTTPException
#     if len(req.question) > 1000:
#         # 过长的输入我们拒绝
#         raise HTTPException(status_code=400, detail="question 太长（>1000 字符）")
#
#     # 返回响应（FastAPI 会用 AskResponse 做序列化与 docs）
#     return AskResponse(session_id=session, answer=answer)

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
