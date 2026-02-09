import dashscope
from http import HTTPStatus
import os
from dotenv import load_dotenv

load_dotenv()
def text_rerank():
    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        query="什么是文本排序模型",
        api_key=os.getenv("QWEN_API_KEY"),
        documents=[
            "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
            "量子计算是计算科学的一个前沿领域",
            "预训练语言模型的发展给文本排序模型带来了新的进展"
        ],
        top_n=10,
        return_documents=True,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )
    if resp.status_code == HTTPStatus.OK:
        print(resp)
    else:
        print(resp)


if __name__ == '__main__':
    text_rerank()