from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- Pydantic 모델 정의 ---

# 여러 Tool에서 공통으로 사용할 소비 항목의 구조를 정의합니다.
class Item(BaseModel):
    category: str = Field(description="소비 카테고리")
    amount: int = Field(description="소비 금액")
    consumption_date: str = Field(description="소비 날짜 (YYYY-MM-DD)", alias='date') # alias for frontend compatibility

# '절약 계획' Tool을 위한 인자 모델
class SavingsPlanArgs(BaseModel):
    items: List[Item] = Field(description="분석할 사용자의 소비 내역 리스트")

# '지출 분석 리포트' Tool을 위한 인자 모델
class SpendingReportArgs(BaseModel):
    start_date: str = Field(description="분석 시작일 (YYYY-MM-DD)")
    end_date: str = Field(description="분석 종료일 (YYYY-MM-DD)")
    items: List[Item] = Field(description="분석할 사용자의 소비 내역 리스트")

# 'DB 조회' Tool을 위한 인자 모델
class SpendingByCategoryArgs(BaseModel):
    year: int = Field(description="조회할 연도(YYYY 형식)")
    month: int = Field(description="조회할 월(1-12 사이의 숫자)")
    category_name: str = Field(description="조회할 소비 카테고리 이름 (예: '식비', '교통', '쇼핑')")

# '퀴즈' Tool을 위한 인자 모델
class QuizArgs(BaseModel):
    items: Optional[List[Item]] = Field(None, description="퀴즈 생성을 위한 소비 내역 리스트 (선택 사항)")

# '기억 기반 답변(RAG)' Tool을 위한 인자 모델 (질문 하나만 받음)
class MemoryQueryArgs(BaseModel):
    question: str = Field(description="사용자의 기억에 대한 질문")


# --- FAISS 벡터 저장소 관리 ---
FAISS_INDEX_PATH = "faiss_index" # 프로젝트 루트에 저장될 폴더 이름
embeddings = OpenAIEmbeddings()

# 가상의 사용자 데이터 (실제로는 DB나 외부 파일에서 로드하는 것이 좋습니다)
USER_MEMOS = [
    "나는 올해 말까지 500만원을 모으는 것이 목표야.",
    "한 달 커피값으로 10만원 이상 쓰지 않기로 약속했어.",
    "여름 휴가 여행을 위해 매달 30만원씩 저축하고 있어.",
]

def get_vector_store():
    """FAISS 인덱스 파일을 로드하거나, 없으면 새로 생성하여 반환합니다."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("INFO: 기존 FAISS 인덱스를 로드합니다.")
        # 로컬 인덱스 로드 시 allow_dangerous_deserialization 필요할 수 있음
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("INFO: 새로운 FAISS 인덱스를 생성하고 저장합니다.")
        vector_store = FAISS.from_texts(texts=USER_MEMOS, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # 가장 유사한 문서 1개 검색