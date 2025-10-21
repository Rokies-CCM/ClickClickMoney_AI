from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from typing import Optional, List, Dict, Any

# .env 파일을 가장 먼저 로드하도록 config 모듈 import
from . import config
from .ai_service import app_graph

app = FastAPI(
    title="ClickClickMoney AI API (v3 Stream)",
    description="기억력 + 스트리밍 기능이 추가된 AI API"
)

# Pydantic 모델 정의
class Item(BaseModel):
    category: str
    amount: int
    consumption_date: str = Field(alias='date')

class ChatRequest(BaseModel):
    user_id: str = Field(..., example="user_1234")
    query: str = Field(..., example="이번 달 과소비 요약 부탁해")
    items: Optional[List[Item]] = Field(None)
    start_date: Optional[str] = Field(None, example="2025-10-01")
    end_date: Optional[str] = Field(None, example="2025-10-30")

# API 엔드포인트 (스트리밍 응답)
@app.post("/api/v1/chat", summary="AI 챗봇 질의응답 (스트리밍)")
async def handle_chat(request: ChatRequest):
    print(f"📨 수신된 요청 | 사용자: {request.user_id}, 질문: {request.query}")
    
    content = request.query
    if request.items:
        items_str = "\n\n[분석할 소비 내역]\n"
        if request.start_date and request.end_date:
            items_str += f"분석 기간: {request.start_date} ~ {request.end_date}\n"
        for item in request.items:
            items_str += f"- {item.consumption_date}, {item.category}, {item.amount:,}원\n"
        content += items_str

    inputs = {"messages": [HumanMessage(content=content)]}
    
    async def stream_generator():
        """LangGraph 스트림을 비동기적으로 처리하고 결과를 SSE 형식으로 전송"""
        last_yielded_value = None
        # 👇 version="v1" 인자를 제거했습니다.
        async for event in app_graph.astream(inputs):
            agent_output = event.get("agent")
            if agent_output:
                messages = agent_output.get("messages")
                if messages:
                    current_content = messages[-1].content
                    if current_content != last_yielded_value and current_content:
                        # SSE 형식으로 데이터 전송
                        yield f"data: {current_content}\n\n"
                        last_yielded_value = current_content
                        # 스트리밍 시각화를 위한 약간의 지연 (선택 사항)
                        await asyncio.sleep(0.02)

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/", summary="Health Check")
def read_root():
    """API 서버 상태를 확인합니다."""
    return {"status": "OK"}