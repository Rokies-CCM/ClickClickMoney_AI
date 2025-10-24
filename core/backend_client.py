# core/backend_client.py
import logging
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger("core.backend_client")
BACKEND_URL = "http://localhost:8080/api"

async def get_consumption_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    page: int = 0,
    size: int = 1000,
    token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """백엔드에서 소비 내역을 가져옵니다."""
    url = f"{BACKEND_URL}/consumptions/load"
    
    params = {"page": page, "size": size}
    
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if category:
        params["category"] = category
    
    headers = {"Content-Type": "application/json"}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            log.info(f"[Backend] GET {url} params={params}")
            response = await client.get(url, params=params, headers=headers)
            
            if response.status_code != 200:
                log.error(f"[Backend] HTTP {response.status_code}")
                return None
            
            data = response.json()
            
            if data and "data" in data:
                count = len(data.get('data', {}).get('content', []))
                log.info(f"[Backend] Fetched {count} records")
            
            return data
            
    except Exception as e:
        log.error(f"[Backend] Error: {e}")
        return None


def format_consumption_for_llm(data: Dict[str, Any]) -> str:
    """백엔드 응답을 텍스트로 변환"""
    if not data or "data" not in data:
        return "소비 데이터를 찾을 수 없습니다."
    
    content = data["data"].get("content", [])
    
    if not content:
        return "해당 기간에 소비 내역이 없습니다."
    
    # 카테고리별 합계
    category_totals = defaultdict(int)
    total_amount = 0
    
    for item in content:
        # categoryName 필드 사용 (로그 확인 결과)
        category = item.get("categoryName") or "기타"
        amount = item.get("amount", 0)
        
        category_totals[category] += int(amount)
        total_amount += int(amount)
    
    # 텍스트 생성
    lines = [
        "[소비 내역 분석]",
        f"총 지출: {total_amount:,}원",
        f"거래 건수: {len(content)}건",
        "",
        "[카테고리별 지출]",
    ]
    
    # 카테고리별로 정렬 (금액 높은 순)
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    for category, amount in sorted_categories:
        percentage = (amount / total_amount * 100) if total_amount > 0 else 0
        lines.append(f"{category}: {amount:,}원 ({percentage:.1f}%)")
    
    # 최근 거래
    lines.append("")
    lines.append("[최근 거래]")
    for item in content[:5]:
        date = item.get("date", "날짜없음")
        category = item.get("categoryName", "기타")
        amount = item.get("amount", 0)
        lines.append(f"{date} | {category} | {amount:,}원")
    
    result = "\n".join(lines)
    log.info(f"[Format] Generated text length: {len(result)}")
    return result


# ========== ✨ 전달 비교 함수 추가 ========== 
async def get_consumption_comparison(
    current_start: str,
    current_end: str,
    token: Optional[str] = None
) -> str:
    """이번 달과 전달 소비 비교 분석"""
    
    # 이번 달 데이터
    current_data = await get_consumption_data(
        start_date=current_start,
        end_date=current_end,
        token=token
    )
    
    # 전달 계산
    current_dt = datetime.strptime(current_start, "%Y-%m-%d")
    prev_month = current_dt.month - 1 if current_dt.month > 1 else 12
    prev_year = current_dt.year if current_dt.month > 1 else current_dt.year - 1
    
    prev_start_dt = datetime(prev_year, prev_month, 1)
    # 전달 마지막 날
    if prev_month == 12:
        prev_end_dt = datetime(prev_year, 12, 31)
    else:
        prev_end_dt = datetime(prev_year, prev_month + 1, 1) - timedelta(days=1)
    
    prev_start = prev_start_dt.strftime("%Y-%m-%d")
    prev_end = prev_end_dt.strftime("%Y-%m-%d")
    
    log.info(f"[Comparison] Current: {current_start} ~ {current_end}")
    log.info(f"[Comparison] Previous: {prev_start} ~ {prev_end}")
    
    # 전달 데이터
    prev_data = await get_consumption_data(
        start_date=prev_start,
        end_date=prev_end,
        token=token
    )
    
    # 데이터 파싱 헬퍼
    def parse_data(data):
        if not data or "data" not in data:
            return {}, 0, 0
        content = data["data"].get("content", [])
        category_totals = defaultdict(int)
        total = 0
        for item in content:
            cat = item.get("categoryName") or "기타"
            amt = int(item.get("amount", 0))
            category_totals[cat] += amt
            total += amt
        return category_totals, total, len(content)
    
    current_cats, current_total, current_count = parse_data(current_data)
    prev_cats, prev_total, prev_count = parse_data(prev_data)
    
    # 비교 텍스트 생성
    lines = [
        "[이번 달 vs 전달 비교]",
        "",
        f"**이번 달 ({current_start[:7]})**",
        f"- 총 지출: {current_total:,}원",
        f"- 거래 건수: {current_count}건",
        "",
        f"**전달 ({prev_start[:7]})**",
        f"- 총 지출: {prev_total:,}원",
        f"- 거래 건수: {prev_count}건",
        "",
    ]
    
    # 증감 분석
    if prev_total > 0:
        diff = current_total - prev_total
        diff_pct = (diff / prev_total) * 100
        
        if diff > 0:
            lines.append(f"💡 **총 지출이 전달 대비 {diff:,}원 증가했어요 (+{diff_pct:.1f}%)**")
        elif diff < 0:
            lines.append(f"💡 **총 지출이 전달 대비 {abs(diff):,}원 감소했어요 ({diff_pct:.1f}%)**")
        else:
            lines.append("💡 **총 지출이 전달과 동일해요**")
    
    lines.append("")
    lines.append("[카테고리별 변화]")
    
    # 카테고리별 증감
    all_cats = set(current_cats.keys()) | set(prev_cats.keys())
    changes = []
    
    for cat in all_cats:
        curr = current_cats.get(cat, 0)
        prev = prev_cats.get(cat, 0)
        diff = curr - prev
        
        if prev > 0:
            diff_pct = (diff / prev) * 100
            changes.append((cat, diff, diff_pct, curr, prev))
        elif curr > 0:
            changes.append((cat, diff, 0, curr, prev))
    
    # 변화량 큰 순으로 정렬
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for cat, diff, diff_pct, curr, prev in changes[:5]:  # 상위 5개만
        if diff > 0:
            lines.append(f"- {cat}: {curr:,}원 (전달 {prev:,}원, +{diff:,}원 ↑{diff_pct:.1f}%)")
        elif diff < 0:
            lines.append(f"- {cat}: {curr:,}원 (전달 {prev:,}원, {diff:,}원 ↓{abs(diff_pct):.1f}%)")
        else:
            lines.append(f"- {cat}: {curr:,}원 (전달과 동일)")
    
    # 이번 달 현황 추가
    lines.append("")
    lines.append("[이번 달 현황]")
    sorted_current = sorted(current_cats.items(), key=lambda x: x[1], reverse=True)
    for cat, amt in sorted_current:
        pct = (amt / current_total * 100) if current_total > 0 else 0
        lines.append(f"{cat}: {amt:,}원 ({pct:.1f}%)")
    
    result = "\n".join(lines)
    log.info(f"[Format] Comparison text length: {len(result)}")
    return result