# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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


# 测试运行：直接执行文件可创建 chat.db
if __name__ == "__main__":
    init_db()
    print("数据库和表创建完成")
