import dashscope
from http import HTTPStatus
from env import *

input_text = "衣服的质量杠杠的"
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=input_text,
    api_key=qwen_api_key
)

if resp.status_code == HTTPStatus.OK:
    # print(type(resp))
    print(resp.output)
    print(resp.output['embeddings'][0]['embedding'])