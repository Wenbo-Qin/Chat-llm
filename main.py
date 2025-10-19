# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from model import *
import uuid

app = FastAPI(title="RAG Q&A Backend (dev)")


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


@app.get("/health")
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

@app.get("/askLLM")
def ask_llm(model_name: str = "deepseek-chat", question: str = "Hi"):
    try:
        answer_text, raw_resp = model_choose(model_name, question)
    except Exception as e:
        # 返回 500 并把错误信息给前端（生产可更弱化错误信息）
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")

    # 2. 返回给前端的 JSON（raw 有时很大或不可序列化，可按需包含）
    payload = {
        "model": model_name,
        "question": question,
        "answer": answer_text,
    }
    if raw_resp is not None:
        payload["raw"] = raw_resp  # 若不需要把 raw 传回前端可移除此行

    return JSONResponse(status_code=200, content=payload["answer"])

