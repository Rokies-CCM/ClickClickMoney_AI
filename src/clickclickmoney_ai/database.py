# src/clickclickmoney_ai/database.py
from langchain_community.utilities import SQLDatabase
from .config import DATABASE_URL  # 👈 같은 패키지 내 config 모듈 import

def get_db_connector() -> SQLDatabase:
    """LangChain의 SQLDatabase 커넥터를 생성합니다."""
    return SQLDatabase.from_uri(DATABASE_URL)