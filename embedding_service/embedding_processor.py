import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
import os

load_dotenv()


def embedding(input_text: str = "衣服的质量杠杠的"):
    try:
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=input_text,
            api_key=os.getenv("QWEN_API_KEY"),
            timeout=10  # 添加 10 秒超时
        )

        if resp.status_code == HTTPStatus.OK:
            return resp.output['embeddings'][0]['embedding']
        else:
            print(f"Embedding failed: {resp.status_code} - {resp.message}")
            return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

if __name__ == "__main__":
    print(embedding())