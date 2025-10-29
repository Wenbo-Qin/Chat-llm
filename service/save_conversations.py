import os
import json
from datetime import datetime


def save_conversation_json(session_id, question, answer_text, model_name):
    """
    将对话记录保存到JSON文件

    Args:
        session_id (str): 会话标识符
        question (str): 用户问题
        answer_text (str): 模型回答
        model_name (str): 模型名称
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 从 service 目录回到根目录

    # 构建 conversations 目录的绝对路径
    conversations_dir = os.path.join(project_root, "conversations")

    # 确保保存目录存在
    os.makedirs(conversations_dir, exist_ok=True)

    # 构造文件路径
    file_path = os.path.join(conversations_dir, f"{session_id}.json")

    # 创建对话记录
    new_record = {
        "timestamp": datetime.now().isoformat(),
        # "integrated_messages": integrated_messages,
        "question": question,
        "answer": answer_text,
        "model": model_name
    }

    # 如果文件存在，读取现有数据并在后面追加新记录
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                conversation_data = json.load(f)
            except json.JSONDecodeError:
                # 如果文件损坏或为空，初始化为列表
                conversation_data = []
    else:
        # 如果是新会话，初始化为空列表
        conversation_data = []

    # 追加新记录
    conversation_data.append(new_record)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
