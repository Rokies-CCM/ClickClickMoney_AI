# src/clickclickmoney_ai/tools/analysis.py
from langchain.tools import StructuredTool
from typing import List

# 같은 tools 패키지 내의 common 모듈에서 모델을 가져옵니다.
from .common import Item, SpendingReportArgs, SavingsPlanArgs

# 👇 함수가 개별 인자를 직접 받도록 시그니처를 수정했습니다.
def generate_spending_report(start_date: str, end_date: str, items: List[Item]) -> str:
    """(AI 기능 1) 종합 지출 분석 리포트를 생성합니다."""
    print(f"TOOL: generate_spending_report 호출됨 ({start_date}~{end_date}, 항목 수: {len(items)}개)")
    if not items: return "분석할 데이터가 없습니다."
    
    category_spending = {}
    total_spending = 0
    for item in items:
        category_spending[item.category] = category_spending.get(item.category, 0) + item.amount
        total_spending += item.amount
        
    top_category = max(category_spending, key=category_spending.get, default="N/A")
    top_category_amount = category_spending.get(top_category, 0)
    report = "## 지출 분석 리포트\n\n"
    report += "### 카테고리별 지출\n"
    for category, amount in sorted(category_spending.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{category}**: {amount:,.0f}원\n"
    report += f"\n**총 지출액**: {total_spending:,.0f}원\n\n"
    report += "### 인사이트\n"
    report += f"이번 분석 기간 동안 **'{top_category}'** 카테고리에서 **{top_category_amount:,.0f}원**으로 가장 많은 금액을 사용하셨습니다.\n\n"
    report += "### 주의\n"
    report += "예산을 초과하지 않도록 주기적으로 지출을 확인하는 습관이 중요합니다."
    return report

# 👇 함수가 개별 인자를 직접 받도록 시그니처를 수정했습니다.
def create_savings_plan(items: List[Item]) -> str:
    """(AI 기능 4) 구체적인 지출 절약 계획을 제안합니다."""
    print(f"TOOL: create_savings_plan 호출됨 (항목 수: {len(items)}개)")
    DISCRETIONARY_CATEGORIES = {"교통", "통신", "쇼핑", "카페/간식", "문화/여가"}
    if not items: return "분석할 소비 내역이 없어 절약 계획을 세울 수 없습니다."
    
    category_spending = {}
    for item in items:
        if item.category in DISCRETIONARY_CATEGORIES:
            category_spending[item.category] = category_spending.get(item.category, 0) + item.amount
            
    if not category_spending: return "줄일 만한 '선택 지출'이 없습니다. 훌륭한 소비 습관을 가지고 계시네요!"
    top_category = max(category_spending, key=category_spending.get)
    top_amount = category_spending[top_category]
    plan = "## 💡 맞춤 절약 계획\n\n"
    plan += f"소비 내역을 보니 **'{top_category}'** 카테고리에서 **{top_amount:,.0f}원**으로 가장 많은 '선택 지출'이 발생했어요.\n\n"
    plan += "**[실천 계획]**\n"
    plan += f"1. **예산 설정**: 다음 달 '{top_category}' 예산을 현재의 80%인 **{int(top_amount * 0.8):,.0f}원**으로 설정해보세요.\n"
    plan += f"2. **소비 기록**: '{top_category}' 관련 지출 발생 시 즉시 기록하여 예산을 관리하세요.\n"
    plan += "3. **대안 찾기**: 카페 지출이 많다면, 일주일에 두 번은 직접 커피를 내려 마시는 습관을 들여보세요.\n\n"
    plan += "작은 실천이 큰 변화를 만듭니다. 응원할게요!"
    return plan

# --- Tool 객체로 변환 ---
# 이 부분은 수정할 필요가 없습니다. LangChain이 args_schema를 보고 알아서 처리합니다.
report_tool = StructuredTool.from_function(
    func=generate_spending_report, name="generate_spending_report",
    description="사용자의 소비 내역과 기간을 받아 종합적인 지출 분석 리포트를 생성합니다.",
    args_schema=SpendingReportArgs
)
savings_plan_tool = StructuredTool.from_function(
    func=create_savings_plan, name="create_savings_plan",
    description="사용자의 소비 내역을 받아 구체적인 절약 계획을 세워줍니다.",
    args_schema=SavingsPlanArgs
)