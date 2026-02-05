"""
测试 ReAct 工作流的对话历史记忆功能
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_conversation_memory():
    """测试对话历史记忆功能"""
    session_id = "test-memory-session"

    print("=" * 60)
    print("测试 ReAct 工作流的对话历史记忆功能")
    print("=" * 60)

    # 第一次对话：告诉模型我的身份
    print("\n第一轮对话：")
    question1 = "记住：我是老师"
    print(f"用户: {question1}")

    response1 = requests.post(
        f"{BASE_URL}/react-ask",
        params={
            "question": question1,
            "session_id": session_id,
            "max_iterations": 5
        },
        timeout=60
    )
    result1 = response1.json()
    print(f"助手: {result1.get('answer', '')[:100]}...")
    print(f"Session ID: {result1.get('session_id')}")

    # 第二轮对话：询问之前的对话
    print("\n第二轮对话：")
    question2 = "刚刚我问了什么？"
    print(f"用户: {question2}")

    response2 = requests.post(
        f"{BASE_URL}/react-ask",
        params={
            "question": question2,
            "session_id": session_id,
            "max_iterations": 5
        },
        timeout=60
    )
    result2 = response2.json()
    answer2 = result2.get('answer', '')
    print(f"助手: {answer2}")
    print(f"Iterations: {result2.get('iteration_count')}")

    # 检查是否成功记住
    if "老师" in answer2 or "记住" in answer2 or question1 in answer2:
        print("\n✅ 测试成功！模型记住了之前的对话内容。")
    else:
        print("\n❌ 测试失败！模型没有记住之前的对话内容。")
        print(f"期望的答案应该包含：'{question1}' 或 '老师'")

    print("=" * 60)

if __name__ == "__main__":
    test_conversation_memory()
