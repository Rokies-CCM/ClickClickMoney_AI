import os
import pandas as pd
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

app = FastAPI(
    title="AI 소비 분석 통합 서비스 (Agent + RAG 버전)",
    description="RAG를 도입하여 더욱 정확한 답변을 제공하는 Agent API",
    version="2.0.0"
)

llm = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model="solar-pro",
    temperature=0.7
)

embeddings = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY,
    model="solar-embedding-1-large"
)

class TransactionItem(BaseModel):
    category: str; amount: float; date: str

class AiRequest(BaseModel):
    question: str; items: List[TransactionItem]

class AiResponse(BaseModel):
    answer: str; insights: Optional[Any] = None

class ChatTip(BaseModel):
    title: str; description: str

class StructuredChatResponse(BaseModel):
    summary: str; actionable_tips: List[ChatTip]; conclusion: str

class Quiz(BaseModel):
    question: str; options: List[str]; answer: int; explanation: str

class QuizList(BaseModel):
    quizzes: List[Quiz]

class SubscriptionAdvice(BaseModel):
    type: str; message: str; potential_savings: float

class SubscriptionAdviceList(BaseModel):
    advices: List[SubscriptionAdvice]

@tool
async def analysis_tool(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    사용자의 전체 소비 내역을 상세히 분석하여 통계, 인사이트, 주의사항을 반환합니다. 
    '가장 많이 쓴', '소비 패턴', '요약', '분석', '인사이트' 등의 단어가 포함된 질문에 사용됩니다.
    """
    if not transactions: return {"error": "분석할 소비 내역이 없습니다."}
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    total_spending = df['amount'].sum()
    cat_dist = df.groupby('category')['amount'].sum().reset_index()
    cat_dist['percentage'] = round((cat_dist['amount'] / total_spending) * 100, 2)
    monthly_trend = df.set_index('date').resample('ME')['amount'].sum().reset_index()
    monthly_trend['month'] = monthly_trend['date'].dt.strftime('%Y-%m')
    
    class Insight(BaseModel): title: str; description: str
    class Warning(BaseModel): title: str; description: str
    class InsightAndWarning(BaseModel): insight: Insight; warning: Warning

    prompt = ChatPromptTemplate.from_template("소비 데이터: {data}\n\n데이터를 바탕으로 인사이트와 주의사항을 각각 하나씩 요약해 주세요.")
    structured_llm = llm.with_structured_output(InsightAndWarning)
    chain = prompt | structured_llm
    llm_response = await chain.ainvoke({"data": cat_dist.to_string()})
    
    return {
        "analysis_summary": "소비 분석 결과입니다.",
        "data": {
            "category_distribution": cat_dist.to_dict('records'),
            "monthly_trend": monthly_trend[['month', 'amount']].to_dict('records'),
            "insight": llm_response.insight.model_dump(),
            "warning": llm_response.warning.model_dump()
        }
    }

@tool
async def quiz_tool() -> Dict[str, Any]:
    """
    사용자에게 금융 및 절약 관련 객관식 퀴즈 3개를 생성하여 반환합니다.
    '퀴즈'라는 단어가 질문에 포함되어 있을 때 사용됩니다.
    """
    prompt = ChatPromptTemplate.from_template("금융 및 절약 상식에 대한 객관식 퀴즈를 서로 다른 주제로 3개 만들어 주세요.")
    structured_llm = llm.with_structured_output(QuizList)
    chain = prompt | structured_llm
    response = await chain.ainvoke({})
    return {
        "quiz_summary": "절약 퀴즈 3개를 생성했습니다.",
        "data": [quiz.model_dump() for quiz in response.quizzes]
    }

# 가맹점 제외해서 제대로 작동 안 함
@tool
async def subscription_analysis_tool(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    사용자의 거래 내역에서 중복되거나 사용하지 않는 정기 결제(구독) 서비스를 찾아 조언을 반환합니다.
    '구독' 또는 '정기결제'라는 단어가 질문에 포함되어 있을 때 사용됩니다.
    """
    prompt = ChatPromptTemplate.from_template("구독 관리 전문가로서, 아래 거래 내역에서 중복/미사용 구독 서비스를 찾아 JSON 리스트 형식으로 조언해주세요.\n\n[거래 내역]\n{data}")
    structured_llm = llm.with_structured_output(SubscriptionAdviceList)
    chain = prompt | structured_llm
    response = await chain.ainvoke({"data": "\n".join([f"- {t['date']} {t['category']}: {t['amount']}원" for t in transactions])})
    return {
        "subscription_summary": "정기 결제 분석 결과입니다.",
        "data": [advice.model_dump() for advice in response.advices]
    }

@tool
async def general_chat_tool(question: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    (RAG 버전) 다른 특정 도구에 해당하지 않는 모든 일반적인 질문에 대해,
    전체 거래 내역에서 가장 관련 있는 정보를 '검색'하여 정확한 답변을 생성하는 기본 도구입니다.
    """
    if not transactions:
        return {
            "chat_summary": "데이터 없음",
            "data": StructuredChatResponse(
                summary="분석할 거래 내역이 없습니다.",
                actionable_tips=[],
                conclusion="먼저 거래 내역을 추가해주세요."
            ).model_dump()
        }

    documents = [f"{t['date']}에 {t['category']} 카테고리에서 {t['amount']}원 사용" for t in transactions]

    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """당신은 사용자의 소비 데이터를 완벽하게 파악하고 있는 친절한 AI 절약 코치입니다.
        아래에 제공된 [관련 거래 내역]을 바탕으로 사용자의 [질문]에 대해 상세하고 정확하게 답변해주세요.
        반드시 JSON 형식으로 구조화된 답변을 생성해야 합니다.

        [관련 거래 내역]
        {context}

        [질문]
        {input}"""
    )
    
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(StructuredChatResponse)
    )

    response = await rag_chain.ainvoke(question)
    
    return {
        "chat_summary": "AI 절약 코치의 답변입니다.",
        "data": response.model_dump()
    }

tools = [analysis_tool, quiz_tool, subscription_analysis_tool, general_chat_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 금융 관련 요청을 해결하는 유능한 AI 비서입니다. 사용자의 질문을 해석하고, 그 의도에 가장 적합한 단 하나의 도구(Tool)를 선택하여 실행해야 합니다."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/llm-request", response_model=AiResponse, tags=["통합 AI 요청 (Agent)"])
async def handle_llm_request_with_agent(request: AiRequest):
    """
    AI Agent가 사용자의 요청을 해석하여, 가장 적합한 Tool을 자율적으로 실행하고 최종 결과를 반환합니다.
    """
    try:
        input_data = f"""사용자 질문: {request.question}

        [참고용 거래 데이터]
        { [item.model_dump() for item in request.items] }
        """

        response = await agent_executor.ainvoke({ "input": input_data })
        final_answer = response['output']
        
        return AiResponse(answer=final_answer, insights=None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Agent 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)