import os
import pandas as pd
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate

dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

app = FastAPI(
    title="AI 소비 분석 통합 서비스",
    description="백엔드의 요청에 따라 지출 분석, 챗봇, 퀴즈 등 다양한 AI 기능을 제공하는 단일 엔드포인트 API",
    version="1.1.0"
)

llm = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model="solar-pro",
    temperature=0.7
)

class TransactionItem(BaseModel):
    category: str
    amount: float
    date: str

class AiRequest(BaseModel):
    question: str
    items: List[TransactionItem]

class AiResponse(BaseModel):
    answer: str
    insights: Optional[Any] = None

class ChatTip(BaseModel):
    title: str = Field(description="실천 가능한 구체적인 절약 팁의 소제목")
    description: str = Field(description="팁에 대한 상세한 설명")

class StructuredChatResponse(BaseModel):
    summary: str = Field(description="사용자의 질문과 데이터에 대한 간략한 상황 요약")
    actionable_tips: List[ChatTip] = Field(description="사용자가 바로 실천할 수 있는 팁 목록")
    conclusion: str = Field(description="사용자를 격려하는 마무리 문장")

class Quiz(BaseModel):
    question: str
    options: List[str]
    answer: int
    explanation: str

class QuizList(BaseModel):
    quizzes: List[Quiz]

# 가맹점 관련 정보가 없어서 지금은 기능하지 않음
"""class SubscriptionAdvice(BaseModel):
    type: str = Field(description="조언의 종류 ('중복 구독' 또는 '미사용 구독')")
    message: str = Field(description="사용자에게 보여줄 조언")
    potential_savings: float = Field(description="이 조언을 따랐을 때 절약 가능한 예상 금액")

class SubscriptionAdviceList(BaseModel):
    advices: List[SubscriptionAdvice]"""

async def _handle_analysis(transactions: List[TransactionItem]) -> Dict[str, Any]:
    """소비 패턴을 분석하고 인사이트/주의사항을 생성"""
    if not transactions:
        return {"error": "분석할 소비 내역이 없습니다."}

    df = pd.DataFrame([t.model_dump() for t in transactions])
    df['date'] = pd.to_datetime(df['date'])
    total_spending = df['amount'].sum()

    cat_dist = df.groupby('category')['amount'].sum().reset_index()
    cat_dist['percentage'] = round((cat_dist['amount'] / total_spending) * 100, 2)
    
    monthly_trend = df.set_index('date').resample('ME')['amount'].sum().reset_index()
    monthly_trend['month'] = monthly_trend['date'].dt.strftime('%Y-%m')
    
    class Insight(BaseModel):
        title: str
        description: str

    class Warning(BaseModel):
        title: str
        description: str

    class InsightAndWarning(BaseModel):
        insight: Insight
        warning: Warning

    prompt = ChatPromptTemplate.from_template(
        """소비 데이터: {data}
        데이터를 바탕으로 사용자의 소비 내역을 전반적으로 통찰하는 인사이트와 개선이 필요한 주의사항을 각각 하나씩 요약해 주세요.""")
    structured_llm = llm.with_structured_output(InsightAndWarning)
    chain = prompt | structured_llm
    
    llm_response = await chain.ainvoke({"data": cat_dist.to_string()})

    return {
        "category_distribution": cat_dist.to_dict('records'),
        "monthly_trend": monthly_trend[['month', 'amount']].to_dict('records'),
        "insight": llm_response.insight.model_dump(),
        "warning": llm_response.warning.model_dump()
    }

# RAG 도입해서 저장한 소비 데이터를 불러오는 형식 고민 중
async def _handle_chat(question: str, transactions: List[TransactionItem]) -> Dict[str, Any]:
    """챗봇 답변 생성"""
    df = pd.DataFrame([t.model_dump() for t in transactions])
    summary = df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()

    prompt = ChatPromptTemplate.from_template(
        """당신은 친절한 AI 절약 코치입니다. 아래 사용자의 최근 소비 데이터 요약과 질문을 바탕으로, 현실적이고 구체적인 답변을 지정된 JSON 형식으로만 생성해주세요.
        반드시 공손한 존댓말을 사용하고 대화 형식의 답변을 해주세요.

        [소비 데이터 요약]
        {summary_data}
        [질문]
        {question}"""
    )
    
    structured_llm = llm.with_structured_output(StructuredChatResponse)
    chain = prompt | structured_llm

    response = await chain.ainvoke({
        "summary_data": summary.to_string(),
        "question": question
    })
    return response.model_dump()

async def _handle_quiz() -> Dict[str, Any]:
    """절약 퀴즈 생성"""
    prompt = ChatPromptTemplate.from_template("금융 및 절약 상식에 대한 객관식 퀴즈를 서로 다른 주제로 3개 만들어 주세요.")
    structured_llm = llm.with_structured_output(QuizList)
    chain = prompt | structured_llm
    response = await chain.ainvoke({})
    return [quiz.model_dump() for quiz in response.quizzes]

# 가맹점 관련 정보가 없어서 지금은 기능하지 않음
"""
async def _handle_subscription_analysis(transactions: List[TransactionItem]) -> List[Dict[str, Any]]:
    "정기 결제 내역 분석 (중복/미사용 탐지)"
    prompt = ChatPromptTemplate.from_template(
        "당신은 구독 관리 전문가입니다. 
        아래 거래 내역 리스트에서 '넷플릭스', '유튜브', '멜론', '스포티파이', '쿠팡' 등 정기 결제로 의심되는 항목들을 찾습니다.
        만약 음악 스트리밍처럼 유사 카테고리의 서비스가 2개 이상 발견되면 '중복 구독' 조언을,
        특정 구독 서비스가 지난 30일간 결제 내역이 없다면 '미사용 구독' 조언을 JSON 리스트 형식으로 만들어줘= 주세요.
        [거래 내역]
        {data}"
    )

    structured_llm = llm.with_structured_output(SubscriptionAdviceList)
    chain = prompt | structured_llm
    response = await chain.ainvoke({
        "data": "\n".join([f"- {t.date} {t.category}: {t.amount}원" for t in transactions])
    })
    return [advice.model_dump() for advice in response.advices]"""

@app.post("/llm-request", response_model=AiResponse, tags=["통합 AI 요청"])
async def handle_llm_request(request: AiRequest):
    """
    백엔드의 모든 AI 관련 요청을 처리합니다.
    `question` 필드에 따라 기능을 라우팅하고, `answer`와 `insights` 필드를 채워 응답합니다.
    """
    question = request.question.lower()
    items = request.items

    try:
        answer_text = ""
        insights_data = None

        """if "구독" in question or "정기결제" in question:
            answer_text = "정기 결제 분석 결과"
            insights_data = await _handle_subscription_analysis(items)"""
        if "퀴즈" in question:
            answer_text = "절약 퀴즈"
            insights_data = await _handle_quiz()
        elif "분석" in question or "요약" in question or "인사이트" in question:
            answer_text = "소비 분석 결과"
            insights_data = await _handle_analysis(items)
        else:
            answer_text = "AI 챗봇"
            insights_data = await _handle_chat(question, items)
        
        return AiResponse(answer=answer_text, insights=insights_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 서버 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)