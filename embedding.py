import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
import os

load_dotenv()


def embedding(input_text: str = "衣服的质量杠杠的"):
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=input_text,
        api_key=os.getenv("QWEN_API_KEY")
    )

    if resp.status_code == HTTPStatus.OK:
        # print(type(resp))
        # print(resp.output)
        # print(resp.output['embeddings'][0]['embedding'])
        return resp.output['embeddings'][0]['embedding']
