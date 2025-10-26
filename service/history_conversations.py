from db import query_messages_by_session_id_with_time_order, query_all_messages, init_db

init_db()  # 确保表已创建


def load_history_conversation(question: str, session_id: str = None):
    history_message = query_messages_by_session_id_with_time_order(session_id)
    if session_id and history_message:
        question = "历史记录：" + history_message.text + "。" + "最新对话：" + question
    return question


if __name__ == "__main__":
    question = load_history_conversation("你好", "0")
    query_all_messages()
