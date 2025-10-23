# core/quant_guard.py
from __future__ import annotations

import re
from typing import Iterable, Set

"""
자연어 수치 가드 규칙 (허용 목록 기반 통과):
- 금액
  1) 300,000원 / 3,000원 / 3000원
  2) ₩300,000 / ￦300,000 / ₩1000 (콤마 없어도 허용)
  3) 20만 원 / 20만원 / 5천 원 / 5천원 / 3억 원 / 3억원
- 퍼센트
  4) 3% / 12.5% / 10~15% / 10 - 15 % / 30 퍼센트 / 20 percent
  5) 퍼센트포인트/기준금리 표기: 0.25%p / 0.25 퍼센트 포인트
  6) 기준금리 인상/인하 자주 쓰는 bp 단위: 25bp / 25 bps
  ※ 전각 퍼센트(％)도 인식
- 빈도(자연어)
  7) 월 3회 / 주 2회 / 월 10일 / 주 5일 / 1주일에 3일 / 연 10회 / 1년 10회 / 주당 3회 / 3회/주 / 월 평균 2회 / 월 평균 2일
※ 단독 숫자(1, 2, 30, 1.)나 목록 번호는 매칭하지 않음.

처리 정책:
- 컨텍스트/출처/사용자 facts 등에서 추출된 "허용 수치"는 그대로 통과
- 허용되지 않은 수치는 다음처럼 자연어로 대체
  * 금액        → "정확한 금액 확인 필요"
  * 비율(%)     → "정확한 비율 확인 필요"
  * 변동폭(%p/bp)→ "정확한 변동폭 확인 필요"
  * 빈도(회/일) → "정확한 횟수 확인 필요"
"""

# ------------------------
# 패턴 정의
# ------------------------

# 공통: 퍼센트 기호(일반/전각)
_PCT = r"(?:%|％)"

# 금액(아라비아+원/통화기호) 또는 한글 단위 금액
_MONEY_ARABIC = r"(?:\d{1,3}(?:,\d{3})+|\d+)\s?원"
_MONEY_SYMBOL = r"(?:[₩￦]\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[₩￦]\s?\d+(?:\.\d+)?)"
_MONEY_KO_UNIT = r"(?:\d+\s?(?:만|천|억)\s?원|\d+(?:만|천|억)원)"

# 퍼센트 (단일/범위 + 한글/영문 표기)
_PERCENT_SINGLE = rf"(?:\d+(?:\.\d+)?\s?{_PCT})"
_PERCENT_RANGE  = rf"(?:\d+(?:\.\d+)?\s*(?:~|-)\s*\d+(?:\.\d+)?\s?{_PCT})"
_PERCENT_WORD   = r"(?:\d+(?:\.\d+)?\s*(?:퍼\s*센\s*트|percent))"

# 퍼센트포인트/기준금리 표기
_PERCENT_POINT  = rf"(?:\d+(?:\.\d+)?\s?%p|\d+(?:\.\d+)?\s?(?:퍼\s*센\s*트\s*포\s*인\s*트))"

# bp/bps (basis points)
_BPS            = r"(?:\d+(?:\.\d+)?\s?bp(?:s)?)"

# 빈도 (월/주/연 + 회/일, 다양한 표현)
_FREQ_1 = r"(?:월\s?\d+(?:회|일))"
_FREQ_2 = r"(?:주\s?\d+(?:회|일))"
_FREQ_3 = r"(?:연\s?\d+회|년\s?\d+회|1년\s?\d+회)"
_FREQ_4 = r"(?:1주일에\s?\d+일|주당\s?\d+회)"
_FREQ_5 = r"(?:\d+회/주|\d+회/월|\d+일/주)"
_FREQ_6 = r"(?:월\s?평균\s?\d+(?:회|일))"

# 추출(허용목록 구성)용 통합 패턴
_RE_NUM = re.compile(
    rf"(?:{_MONEY_ARABIC}|{_MONEY_SYMBOL}|{_MONEY_KO_UNIT}|"
    rf"{_PERCENT_RANGE}|{_PERCENT_SINGLE}|{_PERCENT_WORD}|{_PERCENT_POINT}|{_BPS}|"
    rf"{_FREQ_1}|{_FREQ_2}|{_FREQ_3}|{_FREQ_4}|{_FREQ_5}|{_FREQ_6})",
    re.IGNORECASE | re.UNICODE,
)

# 치환(자연어 대체)용 카테고리별 패턴
_RE_MONEY   = re.compile(rf"(?:{_MONEY_ARABIC}|{_MONEY_SYMBOL}|{_MONEY_KO_UNIT})", re.IGNORECASE | re.UNICODE)
_RE_PERCENT = re.compile(rf"(?:{_PERCENT_RANGE}|{_PERCENT_SINGLE}|{_PERCENT_WORD})", re.IGNORECASE | re.UNICODE)
_RE_PCT_PT  = re.compile(_PERCENT_POINT, re.IGNORECASE | re.UNICODE)
_RE_BPS_RX  = re.compile(_BPS, re.IGNORECASE | re.UNICODE)
_RE_FREQ    = re.compile(rf"(?:{_FREQ_1}|{_FREQ_2}|{_FREQ_3}|{_FREQ_4}|{_FREQ_5}|{_FREQ_6})", re.IGNORECASE | re.UNICODE)

# ------------------------
# 정규화 유틸
# ------------------------

def _canon_percent(s: str) -> str:
    """퍼센트/퍼센트포인트 표기 정규화."""
    t = s.strip().lower()
    t = t.replace("％", "%")  # 전각 → 일반
    # '퍼  센 트' / '퍼센트 포인트' 변형 흡수
    t = re.sub(r"퍼\s*센\s*트\s*포\s*인\s*트", "%p", t)
    t = re.sub(r"퍼\s*센\s*트", "%", t)
    t = t.replace(" percent", "%")
    t = re.sub(r"\s+", " ", t)
    t = t.replace(" %", "%").replace(" %p", "%p")  # 30 % -> 30%, 0.25 %p -> 0.25%p
    # 범위: 10 ~ 15 % -> 10~15%
    t = re.sub(r"\s*~\s*", "~", t)
    t = re.sub(r"\s*-\s*", "-", t)
    t = t.replace("-%", "%")  # 안전장치
    return t

def _canon_bps(s: str) -> str:
    """bp/bps 정규화: 공백 제거, bps -> bp."""
    t = s.strip().lower()
    t = re.sub(r"\s+", "", t)
    t = t.replace("bps", "bp")
    return t

def _canon_money(s: str) -> str:
    """금액 정규화: ' 원' -> '원', 통화기호 공백 제거."""
    t = s.strip()
    t = re.sub(r"\s+원$", "원", t)        # '300,000 원' -> '300,000원'
    t = re.sub(r"([₩￦])\s+", r"\1", t)   # '￦ 300,000' -> '￦300,000'
    t = re.sub(r"\s+", " ", t)
    return t

def _canon_freq(s: str) -> str:
    """빈도 정규화: 불필요 공백 정리."""
    t = re.sub(r"\s+", " ", s.strip())
    return t

def _canon_token(tok: str) -> str:
    """토큰 공통 정규화."""
    lo = tok.lower()
    if ("bp" in lo) or ("bps" in lo):
        return _canon_bps(tok)
    if "%" in tok or "％" in tok or "퍼센트" in lo or "퍼 센 트" in lo or "percent" in lo or "%p" in lo:
        return _canon_percent(tok)
    if "원" in tok or "₩" in tok or "￦" in tok:
        return _canon_money(tok)
    # 회/일 등 빈도
    return _canon_freq(tok)

def _digits_only(s: str) -> str:
    return re.sub(r"[^\d]", "", s or "")

def _core_number(s: str) -> str:
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return m.group(0) if m else ""

# ------------------------
# 공개 API
# ------------------------

def extract_numbers(text: str) -> Set[str]:
    """
    컨텍스트/소스 등에서 허용 수치로 등록할 토큰 추출.
    반환은 '정규화된 토큰' 집합. (표기 차이를 흡수)
    """
    out: Set[str] = set()
    if not text:
        return out
    for m in _RE_NUM.finditer(text):
        tok = m.group(0)
        canon = _canon_token(tok)
        out.add(canon)
        # 허용 범위를 느슨히: 숫자핵/숫자만도 추가
        d = _digits_only(tok)
        if d:
            out.add(d)
        c = _core_number(tok)
        if c:
            out.add(c)
    return out

def _replace_disallowed(text: str, allowed: Set[str], rx: re.Pattern, placeholder: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        can = _canon_token(raw)
        if (can in allowed) or (_digits_only(raw) in allowed) or (_core_number(raw) in allowed):
            return raw
        return placeholder
    return rx.sub(repl, text)

def sanitize_numbers(answer: str, allowed_numbers: Iterable[str]) -> str:
    """
    허용 목록(컨텍스트/소스/사용자 facts/서버 파생 수치)에 있는 수치만 통과.
    그 외는 자연어로 완화:
      - 금액        → "정확한 금액 확인 필요"
      - 비율(%)     → "정확한 비율 확인 필요"
      - 변동폭(%p/bp)→ "정확한 변동폭 확인 필요"
      - 빈도(회/일) → "정확한 횟수 확인 필요"
    """
    if not answer:
        return answer
    allowed_set: Set[str] = set()
    for s in (allowed_numbers or []):
        if not s:
            continue
        c = _canon_token(s)
        allowed_set.add(c)
        d = _digits_only(s)
        if d:
            allowed_set.add(d)
        k = _core_number(s)
        if k:
            allowed_set.add(k)

    t = answer
    # 순서 중요: 변동폭(%p/bp) → 퍼센트(%) → 금액 → 빈도
    t = _replace_disallowed(t, allowed_set, _RE_PCT_PT,  "정확한 변동폭 확인 필요")
    t = _replace_disallowed(t, allowed_set, _RE_BPS_RX,  "정확한 변동폭 확인 필요")
    t = _replace_disallowed(t, allowed_set, _RE_PERCENT, "정확한 비율 확인 필요")
    t = _replace_disallowed(t, allowed_set, _RE_MONEY,   "정확한 금액 확인 필요")
    t = _replace_disallowed(t, allowed_set, _RE_FREQ,    "정확한 횟수 확인 필요")
    return t

__all__ = ["extract_numbers", "sanitize_numbers"]
