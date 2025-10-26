from fastapi import FastAPI
from api.routes import router
import uvicorn
app_chat_llm = FastAPI()

app_chat_llm.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app_chat_llm", host="0.0.0.0", port=8000, reload=True)
