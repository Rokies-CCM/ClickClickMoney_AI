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