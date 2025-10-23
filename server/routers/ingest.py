# server/routers/ingest.py
from __future__ import annotations

import os
import re
import csv
import io
import json
import time
import logging
import calendar
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Set, Optional, Literal

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

# trace 로깅 헬퍼
from server.deps import get_trace_logger, new_trace_id

router = APIRouter()
log = logging.getLogger("ingest")

# ----------------------------------------------------------------------
# 업로드 제한/검증 설정
# ----------------------------------------------------------------------
ALLOWED_CONTENT_TYPES: Set[str] = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
}
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

# ----------------------------------------------------------------------
# 카테고리 매핑 외부화 + 핫리로드 (대분류 / 중분류)
#   - CATEGORY_MAPPING_PATH: JSON 파일 경로 (aliases, buckets)
#   - CATEGORY_HOT_RELOAD_SEC: 변경 체크 주기(초)
# ----------------------------------------------------------------------
DEFAULT_CATEGORY_ALIASES: Dict[str, List[str]] = {
    "생활": ["생활", "생필품", "편의점", "마트", "다이소", "이마트"],
    "식비": ["식비", "식사", "외식", "배달", "카페", "커피", "식자재", "요기요", "배민", "쿠팡이츠"],
    "교통": ["교통", "버스", "지하철", "택시", "카카오T", "주유", "주차", "톨비"],
    "주거": ["월세", "관리비", "전기", "가스", "수도", "청소", "집", "임대료", "전세이자"],
    "통신": ["통신", "인터넷", "휴대폰", "핸드폰", "요금제"],
    "쇼핑": ["쇼핑", "의류", "잡화", "쿠팡", "무신사", "네이버페이", "지그재그"],
    "엔터테인먼트": ["영화", "넷플릭스", "디즈니", "공연", "게임", "취미", "여가"],
    "금융": ["보험", "적금", "대출", "이자", "펀드", "주식", "카드", "수수료"],
    "저축": ["저축", "투자", "비상금", "예비"],
    "기타": ["기타", "선물", "기부", "기타지출", "불명확"],
}

CATEGORY_MAPPING_PATH = os.getenv("CATEGORY_MAPPING_PATH", "data/mapping/category_aliases.json")
CATEGORY_HOT_RELOAD_SEC = int(os.getenv("CATEGORY_HOT_RELOAD_SEC", "60"))

_CATEGORY_ALIASES: Dict[str, List[str]] = dict(DEFAULT_CATEGORY_ALIASES)
_CATEGORY_BUCKETS: Dict[str, List[str]] = {
    "needs": ["생활", "식비", "교통", "주거", "통신", "의료", "교육", "보험"],
    "wants": ["쇼핑", "엔터테인먼트", "여행", "취미", "카페", "기타"],
    "savings": ["저축", "적금", "투자"],
}
_last_mapping_check = 0.0
_last_mapping_mtime = 0.0

def _ensure_category_mapping() -> None:
    """외부 JSON이 존재하면 60초마다 변경 여부를 확인하고 핫리로드."""
    global _CATEGORY_ALIASES, _CATEGORY_BUCKETS, _last_mapping_check, _last_mapping_mtime
    now = time.time()
    if now - _last_mapping_check < CATEGORY_HOT_RELOAD_SEC:
        return
    _last_mapping_check = now

    path = CATEGORY_MAPPING_PATH
    if not path or not os.path.exists(path):
        return
    try:
        mtime = os.path.getmtime(path)
        if mtime <= _last_mapping_mtime:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        aliases = data.get("aliases") or {}
        buckets = data.get("buckets") or {}

        # normalize aliases (각 메인 카테고리를 alias 포함 self 유니온)
        ali_norm: Dict[str, List[str]] = {}
        for main, alias_list in aliases.items():
            items = set(alias_list or [])
            items.add(main)
            ali_norm[main] = sorted(items)

        if ali_norm:
            _CATEGORY_ALIASES = ali_norm
        if isinstance(buckets, dict) and buckets:
            _CATEGORY_BUCKETS = {k: list(v) for k, v in buckets.items()}

        _last_mapping_mtime = mtime
        print(f"[ingest] category mapping reloaded (mtime={int(mtime)})")
    except Exception as e:
        print(f"[ingest] category mapping reload failed: {e}")

def _classify_category(cat_raw: str, memo: str) -> str:
    """문자열에서 대분류 카테고리 추정(외부 매핑 사용)."""
    _ensure_category_mapping()
    text = f"{cat_raw or ''} {memo or ''}".lower()
    for main_cat, aliases in _CATEGORY_ALIASES.items():
        for a in aliases:
            try:
                if (a or '').lower() in text:
                    return main_cat
            except Exception:
                continue
    return "기타"

def _bucket_for(cat: str) -> str:
    """카테고리를 needs|wants|savings로 매핑(없으면 규칙/기본값으로)."""
    for b, cats in _CATEGORY_BUCKETS.items():
        if cat in cats:
            return b
    if any(k in cat for k in ["저축", "투자", "적금"]):
        return "savings"
    return "wants"

# 점심/저녁 빈도 추정 키워드
LUNCH_HINTS = ["점심", "런치", "lunch"]
DINNER_HINTS = ["저녁", "디너", "dinner", "야식"]

WON_TOKEN = re.compile(r"-?\d{1,3}(?:,\d{3})+|\d+")
RE_MONTH = re.compile(r"^\d{4}-\d{2}$")
RE_SAFE_CSV_NAME = re.compile(r"\.csv$", re.IGNORECASE)

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M",
]

# ----------------------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------------------
def _error(status: int, code: str, message: str, trace_id: str) -> ORJSONResponse:
    return ORJSONResponse(status_code=status, content={
        "error": {"code": code, "message": message, "trace_id": trace_id}
    })

def _to_int_amount(s: str) -> int:
    """문자/숫자 혼합 금액 문자열을 정수로 변환 (원 단위 가정)"""
    if not s:
        return 0
    m = WON_TOKEN.findall(str(s))
    if not m:
        return 0
    n = re.sub(r"[^\d\-]", "", m[-1])
    try:
        return int(round(float(n)))
    except Exception:
        return 0

def _count_meals(memo: str) -> Tuple[int, int]:
    m = (memo or "").lower()
    lunch = 1 if any(h in m for h in LUNCH_HINTS) else 0
    dinner = 1 if any(h in m for h in DINNER_HINTS) else 0
    return lunch, dinner

def _fmt(n: int) -> str:
    return f"{n:,}원"

def _validate_headers(fieldnames: List[str]) -> Optional[str]:
    required = ["date", "category", "amount", "memo"]
    if not fieldnames:
        return "CSV 헤더가 비어 있습니다."
    missing = [k for k in required if k not in fieldnames]
    if missing:
        return f"CSV 헤더 누락: {', '.join(missing)} (필수: {', '.join(required)})"
    return None

def _get_days_in_month(month: str) -> int:
    """YYYY-MM → 일수 계산"""
    try:
        y, m = map(int, month.split("-"))
        return calendar.monthrange(y, m)[1]
    except Exception:
        return 30

def _parse_date(s: str) -> Optional[datetime]:
    """여러 포맷+한국어 표기를 지원하는 날짜 파서"""
    t = (s or "").strip()
    if not t:
        return None
    # 한국어 2025년 10월 21일
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    # 느슨한 yyyy.mm.dd / yyyy-mm-dd
    m2 = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", t)
    if m2:
        try:
            return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except Exception:
            return None
    return None

def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{dt.year}-W{iso.week:02d}"

def _weekday_name(dt: datetime) -> str:
    # 0=월 ... 6=일
    names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return names[dt.weekday()]

@dataclass
class Row:
    dt: datetime
    category: str
    amount: int
    memo: str

# ----------------------------------------------------------------------
# 집계/분석 로직
# ----------------------------------------------------------------------
def _aggregate(rows: List[Row]) -> Dict[str, List[Dict]]:
    """월/주/요일 집계"""
    monthly: Dict[str, int] = {}
    weekly: Dict[str, int] = {}
    weekday_sum: Dict[str, int] = {}
    for r in rows:
        mkey = r.dt.strftime("%Y-%m")
        wkey = _week_key(r.dt)
        dkey = _weekday_name(r.dt)
        monthly[mkey] = monthly.get(mkey, 0) + r.amount
        weekly[wkey] = weekly.get(wkey, 0) + r.amount
        weekday_sum[dkey] = weekday_sum.get(dkey, 0) + r.amount
    monthly_list = sorted([{"month": k, "total": v} for k, v in monthly.items()], key=lambda x: x["month"])
    weekly_list = sorted([{"week": k, "total": v} for k, v in weekly.items()], key=lambda x: x["week"])
    weekday_list = [{"weekday": k, "total": weekday_sum.get(k, 0)} for k in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]]
    return {"monthly": monthly_list, "weekly": weekly_list, "weekday": weekday_list}

def _top_categories(rows: List[Row], n: int = 5) -> List[Dict]:
    cs: Dict[str, int] = {}
    for r in rows:
        cs[r.category] = cs.get(r.category, 0) + r.amount
    return sorted([{"category": k, "total": v} for k, v in cs.items()], key=lambda x: x["total"], reverse=True)[:n]

def _mom(all_rows: List[Row], target_month: Optional[str]) -> Optional[Dict]:
    """전월 대비 (전체 데이터 기반에서 target_month가 있으면 그 월 기준, 없으면 마지막 두 달)"""
    monthly: Dict[str, int] = {}
    for r in all_rows:
        k = r.dt.strftime("%Y-%m")
        monthly[k] = monthly.get(k, 0) + r.amount
    if not monthly:
        return None
    months_sorted = sorted(monthly.keys())
    if target_month and target_month in monthly:
        idx = months_sorted.index(target_month)
        if idx == 0:
            return None
        prev_m, cur_m = months_sorted[idx-1], target_month
    else:
        if len(months_sorted) < 2:
            return None
        prev_m, cur_m = months_sorted[-2], months_sorted[-1]
    prev_v, cur_v = monthly.get(prev_m, 0), monthly.get(cur_m, 0)
    delta = cur_v - prev_v
    pct = None if prev_v == 0 else round(delta / prev_v * 100, 2)
    return {
        "base_month": prev_m,
        "current_month": cur_m,
        "base_total": prev_v,
        "current_total": cur_v,
        "delta": delta,
        "pct": pct,
        "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
    }

def _limits_for_month(rows: List[Row], cut_ratio: int = 10) -> List[Dict]:
    """선택된 월 데이터 기준 Top3 카테고리 -cut_ratio% 제안"""
    cs: Dict[str, int] = {}
    for r in rows:
        cs[r.category] = cs.get(r.category, 0) + r.amount
    top3 = sorted([{"category": k, "total": v} for k, v in cs.items()], key=lambda x: x["total"], reverse=True)[:3]
    out = []
    for it in top3:
        cur = it["total"]
        suggested = max(0, int(round(cur * (100 - cut_ratio) / 100)))
        out.append({
            "category": it["category"],
            "current": cur,
            "suggested": suggested,
            "current_display": _fmt(cur),
            "suggested_display": _fmt(suggested),
            "cut_ratio": cut_ratio
        })
    return out

# ----------------------------------------------------------------------
# 메인 라우트 - CSV 인입
# ----------------------------------------------------------------------
@router.post("/ingest/ledger")
async def ingest_ledger(
    file: UploadFile = File(..., description="CSV 파일 (헤더 예: date,category,amount,memo)"),
    month: Optional[str] = Form(None, description="YYYY-MM 형식(선택)"),
):
    """
    사용자의 가계부 CSV를 분석하여 카테고리별 합계, 상위 소비 항목,
    월/주/요일 집계, 전월대비, 한도 제안, 점심/저녁 빈도 등을 반환합니다.
    """
    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)

    try:
        # 업로드 검증
        if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
            return _error(400, "INVALID_CONTENT_TYPE", f"CSV만 허용됩니다. ({file.content_type})", trace_id)
        if file.filename and not RE_SAFE_CSV_NAME.search(file.filename):
            return _error(400, "INVALID_FILENAME", "파일 확장자가 .csv 이어야 합니다.", trace_id)

        blob = await file.read()
        if not blob:
            return _error(400, "EMPTY_FILE", "빈 파일입니다.", trace_id)
        if len(blob) > MAX_UPLOAD_BYTES:
            return _error(400, "FILE_TOO_LARGE", f"최대 {MAX_UPLOAD_MB}MB까지 허용됩니다.", trace_id)

        # 디코딩
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = blob.decode("utf-8", errors="ignore")

        # month 형식 검증
        if month and not RE_MONTH.match(month):
            return _error(400, "INVALID_MONTH", "month는 YYYY-MM 형식이어야 합니다.", trace_id)

        # CSV 파싱
        reader = csv.DictReader(io.StringIO(text))
        err = _validate_headers(reader.fieldnames or [])
        if err:
            return _error(400, "INVALID_HEADERS", err, trace_id)

        # 전체 행 파싱 (all_rows) — 날짜/금액 유효성 검사
        all_rows: List[Row] = []
        row_errors: List[str] = []
        total_rows = 0
        lunch_cnt_all = dinner_cnt_all = 0

        for idx, row in enumerate(reader, start=2):
            total_rows += 1
            dt = _parse_date(row.get("date", ""))
            if dt is None:
                row_errors.append(f"row {idx}: invalid date")
                continue
            amt = _to_int_amount(row.get("amount") or "")
            if amt <= 0:
                # 수입/환급(음수) 또는 0원은 제외(정책 필요시 변경)
                continue
            memo = (row.get("memo") or "").strip()
            cat_main = _classify_category((row.get("category") or "").strip(), memo)

            all_rows.append(Row(dt=dt, category=cat_main, amount=amt, memo=memo))
            lc, dc = _count_meals(memo)
            lunch_cnt_all += lc
            dinner_cnt_all += dc

        if not all_rows:
            return _error(400, "NO_VALID_ROWS", "유효한 데이터가 없습니다. (date/amount 확인)", trace_id)

        # 선택 월 필터 (없으면 전체)
        if month:
            selected_rows = [r for r in all_rows if r.dt.strftime("%Y-%m") == month]
        else:
            selected_rows = list(all_rows)

        if not selected_rows:
            return _error(400, "NO_ROWS_FOR_MONTH", f"{month}에 해당하는 데이터가 없습니다.", trace_id)

        # 집계/요약
        summary = _aggregate(selected_rows)
        topN = _top_categories(selected_rows, n=5)

        # 전월 대비 (전체 데이터 기준에서 대상월을 지정)
        mom = _mom(all_rows, target_month=month)

        # 한도 제안 (선택월 기준)
        limits = _limits_for_month(selected_rows, cut_ratio=10)

        # 식사 빈도(선택월 기준 재계산)
        lunch_cnt_sel = sum(_count_meals(r.memo)[0] for r in selected_rows)
        dinner_cnt_sel = sum(_count_meals(r.memo)[1] for r in selected_rows)

        # 총합/기간
        sum_total = sum(r.amount for r in selected_rows)
        min_d = min(r.dt for r in selected_rows).date()
        max_d = max(r.dt for r in selected_rows).date()

        # 결과 facts (chat에 바로 넘길 요약)
        facts: Dict[str, str] = {}
        facts["기간"] = f"{min_d} ~ {max_d}"
        facts["총지출"] = _fmt(sum_total)
        if mom:
            s = f"{mom['current_month']} 총지출 {_fmt(mom['current_total'])}"
            if mom["pct"] is not None:
                s += f" (전월 대비 {mom['pct']}%)"
            facts["전월대비"] = s
        # Top3 + 한도 제안
        for i, item in enumerate(sorted(topN, key=lambda x: x["total"], reverse=True)[:3], start=1):
            facts[f"Top{i} {item['category']}"] = _fmt(item["total"])
        for lim in limits:
            facts[f"{lim['category']} 한도 제안"] = lim["suggested_display"]
        # 식사 빈도
        if lunch_cnt_sel > 0:
            facts["점심빈도"] = f"월 {lunch_cnt_sel}일"
        if dinner_cnt_sel > 0:
            facts["저녁빈도"] = f"월 {dinner_cnt_sel}일"

        if sum_total > 0:
            for item in sorted(topN, key=lambda x: x["total"], reverse=True)[:3]:
                share = round(item["total"] / sum_total * 100, 1)
                facts[f"{item['category']} 비중"] = f"{share}%"
        
        # 표시용 Top카드
        top_display = [{"category": x["category"], "total": x["total"], "display": _fmt(x["total"])} for x in topN]

        # period 블록
        period = {
            "min_date": str(min_d),
            "max_date": str(max_d),
            "row_errors": row_errors[:10],  # 과다 노출 방지
        }

        # info (기존 호환 필드 유지 + 확장)
        sum_by_category: Dict[str, int] = {}
        for r in selected_rows:
            sum_by_category[r.category] = sum_by_category.get(r.category, 0) + r.amount
        top_category = max(sum_by_category, key=sum_by_category.get) if sum_by_category else None
        avg_daily = round(sum_total / _get_days_in_month(month), 1) if (month and sum_total > 0) else None

        info = {
            "month": month,
            "sum_total": sum_total,
            "sum_by_category": sum_by_category,
            "top_spend_category": top_category,
            "avg_daily_spend": avg_daily,
            "lunch_days": lunch_cnt_sel,
            "dinner_days": dinner_cnt_sel,
            "rows": len(selected_rows),
        }

        tlog.info(f"/ingest/ledger ok rows_all={total_rows} rows_used={len(selected_rows)} month={month} total={sum_total} top={top_category}")

        # 응답 반환(기존 + 확장 동시 제공)
        return ORJSONResponse(
            content={
                "ok": True,
                "trace_id": trace_id,
                "items": len(selected_rows),
                "period": period,
                "summary": summary,                 # 월/주/요일 집계
                "top_categories": top_display,      # Top N (표시형)
                "mom": mom,                         # 전월 대비
                "limits": limits,                   # 한도 제안
                "facts": facts,                     # /v1/chat facts로 바로 사용
                # 하위호환 정보 블록
                "info": info,
            }
        )

    except Exception as e:
        tlog.exception(f"/ingest/ledger failed: {e}")
        return _error(500, "INGEST_ERROR", "업로드 처리 중 오류가 발생했습니다.", trace_id)

# ----------------------------------------------------------------------
# 예산안 API — /v1/ingest/ledger/budget
# ----------------------------------------------------------------------
class BudgetPayload(BaseModel):
    sum_by_category: Dict[str, float] = Field(..., description="카테고리별 지출 합계 (원)")
    income: Optional[float] = Field(default=None, description="월 소득(선택)")
    strategy: Literal["50_30_20", "zero_base", "last_month_avg"] = Field(default="50_30_20")

class BudgetItem(BaseModel):
    current: int
    target: int
    diff: int
    action: Literal["cut", "grow", "keep"]
    ratio_current: float | None = None
    ratio_target: float | None = None
    bucket: Optional[str] = None

class BudgetResponse(BaseModel):
    strategy: str
    budget_total: int
    total_spend: int
    savings_target: int
    by_category: Dict[str, BudgetItem]

def _normalize_sum_by_category(raw: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            n = int(round(float(v)))
        except Exception:
            n = 0
        cat = _classify_category(k, "")
        out[cat] = out.get(cat, 0) + max(0, n)
    return out

def _distribute(total: float, shares: Dict[str, float]) -> Dict[str, int]:
    s = sum(shares.values()) or 1.0
    alloc = {k: int(round(total * (v / s))) for k, v in shares.items()}
    # rounding fix
    delta = int(round(total)) - sum(alloc.values())
    if delta != 0 and alloc:
        k = max(alloc, key=lambda x: alloc[x])
        alloc[k] += delta
    return alloc

def _build_budget(strategy: str, sums: Dict[str, int], income: Optional[float]) -> BudgetResponse:
    total_spend = sum(max(0, v) for v in sums.values())
    budget_total = int(round(income)) if income and income > 0 else total_spend
    if strategy == "50_30_20":
        needs_total = int(round(budget_total * 0.50))
        wants_total = int(round(budget_total * 0.30))
        savings_target = int(round(budget_total * 0.20))

        needs_cats = {k: v for k, v in sums.items() if _bucket_for(k) == "needs"}
        wants_cats = {k: v for k, v in sums.items() if _bucket_for(k) == "wants"}

        needs_alloc = _distribute(needs_total, needs_cats or {"생활": 1, "식비": 1, "교통": 1, "주거": 1})
        wants_alloc = _distribute(wants_total, wants_cats or {"쇼핑": 1, "엔터테인먼트": 1, "기타": 1})

        by_cat: Dict[str, BudgetItem] = {}
        all_cats = set(list(sums.keys()) + list(needs_alloc.keys()) + list(wants_alloc.keys()))
        for cat in all_cats:
            cur = int(sums.get(cat, 0))
            tgt = int(needs_alloc.get(cat, 0) + wants_alloc.get(cat, 0))
            bc = _bucket_for(cat)
            d = tgt - cur
            act = "cut" if d < 0 else ("grow" if d > 0 else "keep")
            rc = (cur / budget_total) if budget_total else None
            rt = (tgt / budget_total) if budget_total else None
            by_cat[cat] = BudgetItem(current=cur, target=tgt, diff=d, action=act,
                                     ratio_current=rc, ratio_target=rt, bucket=bc)

        # 저축/투자 보강
        if "저축" not in by_cat:
            by_cat["저축"] = BudgetItem(current=0, target=savings_target, diff=savings_target, action="grow",
                                        ratio_current=0.0, ratio_target=(savings_target / budget_total) if budget_total else 0.0,
                                        bucket="savings")
        else:
            item = by_cat["저축"]
            item.target = savings_target
            item.diff = savings_target - item.current
            item.action = "cut" if item.diff < 0 else ("grow" if item.diff > 0 else "keep")
            item.ratio_target = (savings_target / budget_total) if budget_total else 0.0

        return BudgetResponse(strategy=strategy, budget_total=budget_total, total_spend=total_spend,
                              savings_target=savings_target, by_category=by_cat)

    elif strategy == "zero_base":
        savings_target = int(round(budget_total * 0.20))
        by_cat: Dict[str, BudgetItem] = {}
        remaining = budget_total - savings_target
        shares = {}
        for cat, cur in sums.items():
            factor = 0.80 if _bucket_for(cat) == "needs" else 0.60
            tgt = int(round(max(0, cur) * factor))
            shares[cat] = tgt
        total_tgt = sum(shares.values()) or 1
        scale = remaining / total_tgt if total_tgt else 1.0
        for cat, base_tgt in shares.items():
            cur = int(sums.get(cat, 0))
            tgt = int(round(base_tgt * scale))
            d = tgt - cur
            act = "cut" if d < 0 else ("grow" if d > 0 else "keep")
            rc = (cur / budget_total) if budget_total else None
            rt = (tgt / budget_total) if budget_total else None
            by_cat[cat] = BudgetItem(current=cur, target=tgt, diff=d, action=act,
                                     ratio_current=rc, ratio_target=rt, bucket=_bucket_for(cat))
        if "저축" not in by_cat:
            by_cat["저축"] = BudgetItem(current=0, target=savings_target, diff=savings_target, action="grow",
                                        ratio_current=0.0, ratio_target=(savings_target / budget_total) if budget_total else 0.0,
                                        bucket="savings")
        return BudgetResponse(strategy=strategy, budget_total=budget_total, total_spend=total_spend,
                              savings_target=savings_target, by_category=by_cat)

    else:  # last_month_avg (데이터 없으면 현재 유지)
        savings_target = int(round(budget_total * 0.15))
        by_cat: Dict[str, BudgetItem] = {}
        for cat, cur in sums.items():
            by_cat[cat] = BudgetItem(current=cur, target=cur, diff=0, action="keep",
                                     ratio_current=(cur / budget_total) if budget_total else None,
                                     ratio_target=(cur / budget_total) if budget_total else None,
                                     bucket=_bucket_for(cat))
        if "저축" not in by_cat:
            by_cat["저축"] = BudgetItem(current=0, target=savings_target, diff=savings_target, action="grow",
                                        ratio_current=0.0, ratio_target=(savings_target / budget_total) if budget_total else 0.0,
                                        bucket="savings")
        return BudgetResponse(strategy=strategy, budget_total=budget_total, total_spend=total_spend,
                              savings_target=savings_target, by_category=by_cat)

@router.post("/ingest/ledger/budget", response_model=BudgetResponse, summary="카테고리별 예산안 생성")
async def ingest_ledger_budget(payload: BudgetPayload):
    """
    전략을 선택해 카테고리별 권장 한도를 계산합니다.

    - **strategy**: `50_30_20` | `zero_base` | `last_month_avg`
    - **sum_by_category**: 예) {"식비": 350000, "주거": 550000, "쇼핑": 120000}
    - **income**(선택): 월 소득. 없으면 현재 총지출을 예산 총액으로 사용.

    반환:
    - `by_category`에 카테고리별 `current`, `target`, `diff`, `action`, `ratio_*`, `bucket`
    """
    sums = _normalize_sum_by_category(payload.sum_by_category or {})
    resp = _build_budget(payload.strategy, sums, payload.income)
    return resp
