# src/clickclickmoney_ai/config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 할당
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DB_URI")

# 필수 환경 변수 확인
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")