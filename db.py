# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

# 1. 数据库 URL（SQLite 文件）
DATABASE_URL = "sqlite:///./chat.db"

# 2. 创建数据库引擎
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# 3. 创建 session
SessionLocal = sessionmaker(bind=engine)

# 4. 创建基类
Base = declarative_base()

# 5. 定义 Message 表
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String, default="user")
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# 6. 初始化数据库（创建表）
def init_db():
    Base.metadata.create_all(bind=engine)


def save_conversation_sql(session_id: str, question: str, answer_text: str, model_name: str = "deepseek-chat"):
    """
    将对话记录保存到数据库 (新增的SQLAlchemy方法)
    
    Args:
        session_id (str): 会话标识符
        question (str): 用户问题
        answer_text (str): 模型回答
        model_name (str): 模型名称
    
    Returns:
        bool: 保存成功返回 True，否则返回 False
    """
    db = SessionLocal()
    try:
        # 保存用户问题
        user_message = Message(
            session_id=session_id,
            role="user",
            text=question,
            created_at=datetime.datetime.utcnow()
        )
        db.add(user_message)

        # 保存模型回答
        ai_message = Message(
            session_id=session_id,
            role="assistant",
            text=answer_text,
            created_at=datetime.datetime.utcnow()
        )
        db.add(ai_message)

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"保存对话记录时出错: {e}")
        return False
    finally:
        db.close()


def query_all_messages():
    db = SessionLocal()
    try:
        messages = db.query(Message).all()
        for msg in messages:
            print(f"Session: {msg.session_id}")
            print(f"Role: {msg.role}")
            print(f"text: {msg.text}")
            print(f"created_at: {msg.created_at}")
            print("-" * 30)
    finally:
        db.close()


# 测试运行：直接执行文件可创建 chat.db
if __name__ == "__main__":
    query_all_messages()
