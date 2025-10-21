import os
import pymysql
from langchain.tools import StructuredTool
from pydantic import BaseModel
from typing import List, Optional

# LangChain 및 AI 모델 관련 import 추가
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 같은 tools 패키지 내의 common 모듈에서 모델 및 retriever 가져오기
from .common import SpendingByCategoryArgs, QuizArgs, MemoryQueryArgs, retriever, Item

# --- 도구 함수 정의 ---

def get_total_spending_by_category(year: int, month: int, category_name: str) -> str:
    """(AI 기능 2) DB에서 특정 카테고리 지출액을 조회합니다."""
    print(f"TOOL: get_total_spending_by_category 호출됨 (인자: {year}, {month}, {category_name})")
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"), user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD"), db=os.getenv("DB_NAME", "clickclickmoney"),
        charset='utf8'
    )
    query = "SELECT SUM(c.amount) FROM consumption c JOIN categories cat ON c.category_id = cat.id WHERE YEAR(c.consumption_date) = %s AND MONTH(c.consumption_date) = %s AND cat.name = %s"
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (year, month, category_name))
            result = cursor.fetchone()[0]
            if result is None: return f"{year}년 {month}월의 '{category_name}' 카테고리 지출 내역이 없습니다."
            else: return f"{year}년 {month}월 '{category_name}' 카테고리의 총 지출액은 {int(result):,}원입니다."
    except Exception as e:
        print(f"DB 조회 오류: {e}")
        return "데이터베이스 조회 중 오류가 발생했습니다."
    finally:
        conn.close()

def create_finance_quiz(items: Optional[List[Item]] = None) -> str:
    """(AI 기능 3) 금융 퀴즈를 생성합니다."""
    print("TOOL: create_finance_quiz 호출됨")
    if items:
        top_item = max(items, key=lambda x: x.amount, default=None)
        if top_item and top_item.category == '쇼핑':
            return "## 🛍️ 맞춤 퀴즈!\n\n최근 쇼핑에 큰 금액을 사용하셨네요! 충동구매를 줄이는 가장 효과적인 방법은 무엇일까요?\n1. 세일 기간 노리기\n2. 구매 전 24시간 기다리기\n3. 스트레스 해소용 쇼핑\n\n정답은 **2번**! 구매 전 잠시 생각하면 불필요한 지출을 크게 줄일 수 있습니다."
    return "## 💰 금융 퀴즈!\n\n금리가 오르면 일반적으로 예금 보유자에게는 어떤 영향이 있을까요?\n1. 불리하다\n2. 유리하다\n3. 영향 없다\n\n정답은 **2번**! 예금 이자가 늘어나기 때문에 유리합니다."

def answer_from_memory(question: str) -> str:
    """(기억력 AI) 사용자의 개인적인 목표나 약속에 대한 질문에 답변합니다."""
    print(f"TOOL: answer_from_memory 호출됨 (질문: {question})")
    
    retrieved_docs = retriever.invoke(question)
    
    if not retrieved_docs:
        return "죄송하지만, 해당 질문과 관련된 정보가 제 기억에 없습니다."
        
    prompt = ChatPromptTemplate.from_template("""
    당신은 친절한 재무 코치입니다. 아래의 기억을 바탕으로 사용자의 질문에 답변해주세요.

    기억:
    {context}

    질문: {input}
    """)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({
        "input": question,
        "context": retrieved_docs
    })
    return result

# --- Tool 객체로 변환 ---
db_tool = StructuredTool.from_function(
    func=get_total_spending_by_category, name="get_total_spending_by_category",
    description="사용자의 월별 특정 카테고리 소비 총액을 조회합니다.",
    args_schema=SpendingByCategoryArgs
)
quiz_tool = StructuredTool.from_function(
    func=create_finance_quiz, name="create_finance_quiz",
    description="사용자에게 금융 관련 퀴즈를 내줍니다.",
    args_schema=QuizArgs
)
rag_tool = StructuredTool.from_function(
    func=answer_from_memory,
    name="answer_from_user_memory",
    description="사용자의 재무 목표, 과거의 약속 등 개인적인 정보와 관련된 질문에 답변할 때 사용됩니다. 예를 들어 '내 목표가 뭐였지?' 같은 질문에 사용합니다.",
    args_schema=MemoryQueryArgs # Pydantic 모델 사용
)