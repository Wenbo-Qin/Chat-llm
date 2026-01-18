from logging import log
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to different LLM models based on user choice
def openai_chat():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-5",
        input="Write a one-sentence bedtime story about a unicorn."
    )
    log.debug("response", response)
    return response

def deepseek_chat(model: str, user_content: str):
    """
    发送 single-turn 聊天请求到 Deepseek（同步方式）。
    返回 (text, raw_response_safe_dict_or_None)
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_content},
        ],
        stream=False
    )

    # 从 SDK 响应里提取文本（按你示例的结构）
    try:
        text = resp.choices[0].message.content
    except Exception:
        # 若 SDK 返回结构不同，保护性返回整个 resp 的字符串表示
        text = getattr(resp, "text", None) or str(resp)

    # 尝试把原始响应序列化为 dict（如果 SDK 支持 to_dict 方法）
    raw = None
    if hasattr(resp, "to_dict"):
        try:
            raw = resp.to_dict()
        except Exception:
            raw = None

    return text, raw


def model_choose(model_name: str, user_content: str):
    if model_name == "deepseek-chat":
        return deepseek_chat("deepseek-chat", user_content)
    elif model_name == "deepseek-reasoner":
        return deepseek_chat("deepseek-reasoner", user_content)
    elif model_name == "openai":
        return openai_chat()
    else:
        print("model not found")

# test model
if __name__ == "__main__":
    model_choose("deepseek-chat", "你好")
