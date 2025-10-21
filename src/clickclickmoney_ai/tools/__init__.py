# 각 모듈에서 Tool 객체들을 가져옵니다.
from .analysis import report_tool, savings_plan_tool
from .coaching import db_tool, quiz_tool, rag_tool

# 모든 Tool 객체를 하나의 리스트로 통합합니다.
# AI 에이전트는 이 리스트를 보고 사용할 수 있는 도구들을 파악합니다.
all_tools = [
    report_tool,
    savings_plan_tool,
    db_tool,
    quiz_tool,
    rag_tool,
]