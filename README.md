# ai-chatbot

고성능/저지연 AI 챗봇 (FastAPI + LangChain 0.3.x + FAISS + Groq/OpenAI + Tavily + Upstage Re-ranker)

## Quick Start (로컬)
```bash
poetry lock --no-cache && poetry install
# 문서 준비
mkdir -p data/docs
echo "프로젝트 개요 텍스트입니다." > data/docs/guide.txt
# 인덱싱
poetry run python scripts/build_index.py
# 서버
poetry run uvicorn server.main:app --host 0.0.0.0 --port 8000
