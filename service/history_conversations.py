from service.db import query_messages_by_session_id_with_time_order, query_all_messages, init_db

init_db()  # 确保表已创建


def load_history_conversation(question: str, session_id: str = None):
    history_message = query_messages_by_session_id_with_time_order(session_id)
    if session_id and history_message:
        question = "AI最近一条回复：" + history_message.text + "。" + "最新对话：" + question
    return question


if __name__ == "__main__":
    # 测试历史消息查询
    question = load_history_conversation("你好", "dev-test")
    print(question)
    query_all_messages()
