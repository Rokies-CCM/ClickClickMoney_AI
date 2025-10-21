# core/metrics.py
import time
import statistics
from typing import List, Dict, Any

_bench_cache: List[Dict[str, Any]] = []

def _num(x, default=0.0) -> float:
    """
    숫자 변환 유틸: None/''/비숫자에도 안전.
    default가 None이면 0.0을 사용.
    """
    safe_default = 0.0 if default is None else default
    if x is None:
        return float(safe_default)
    try:
        return float(x)
    except Exception:
        return float(safe_default)

def _int(x, default=0) -> int:
    """
    int 변환 유틸: None/''/비숫자에도 안전.
    """
    if x is None:
        return int(default)
    try:
        return int(float(x))
    except Exception:
        return int(default)

def _sanitize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """
    대시보드가 기대하는 필드를 항상 채운다.
    (없으면 기본값으로 보정) → 렌더 크래시/요약 계산 오류 방지
    """
    out = dict(e or {})
    out.setdefault("provider", "unknown")
    out.setdefault("model", "-")
    out.setdefault("q", "")

    # 시간(ms)
    rt = _num(out.get("retrieval_ms"), 0.0)
    gt = _num(out.get("generation_ms"), 0.0)
    tt_raw = out.get("total_ms", None)
    tt = _num(tt_raw, None)

    if tt_raw is None:
        # total이 없으면 합으로 보정
        tt = rt + gt
    else:
        # total이 있어도 음수/NaN 방지
        tt = _num(tt_raw, rt + gt)

    out["retrieval_ms"] = rt
    out["generation_ms"] = gt
    out["total_ms"] = tt

    # 토큰
    out["tokens_in"]  = _int(out.get("tokens_in"), 0)
    out["tokens_out"] = _int(out.get("tokens_out"), 0)

    # 점수
    if "rougeL" not in out:
        out["rougeL"] = None

    # trace_id 옵션
    out.setdefault("trace_id", None)

    return out

def append_bench_cache(entry: Dict[str, Any]):
    """LiveTest 결과를 실시간 메모리에 추가 (안전 보정 포함)"""
    entry = _sanitize_entry(entry)
    entry["timestamp"] = time.time()
    _bench_cache.append(entry)
    if len(_bench_cache) > 100:
        _bench_cache.pop(0)  # 오래된 데이터 삭제

def get_bench_summary() -> Dict[str, Any]:
    if not _bench_cache:
        return {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "avg_ms": 0.0,
            "avg_tokens_in": 0.0,
            "avg_tokens_out": 0.0,
            "avg_rougeL": None,
            "avg_bleu": None,
            "avg_evidence": None,
            "provider_avg_ms": {},
        }

    sanitized = [_sanitize_entry(r) for r in _bench_cache]

    lat = [r["total_ms"] for r in sanitized]
    tokens_in  = [r["tokens_in"]  for r in sanitized]
    tokens_out = [r["tokens_out"] for r in sanitized]
    rouge = [r.get("rougeL") for r in sanitized if r.get("rougeL") is not None]
    bleu  = [r.get("bleu")  for r in sanitized if r.get("bleu")  is not None]
    evd   = [r.get("evidence_score") for r in sanitized if r.get("evidence_score") is not None]

    from collections import defaultdict
    tmp = defaultdict(list)
    for r in sanitized:
        prov = r.get("provider") or "unknown"
        tmp[prov].append(r["total_ms"])
    prov_lat = {k: round(sum(v)/len(v), 1) for k, v in tmp.items() if v}

    try:
        p50 = statistics.median(lat) if lat else 0.0
    except Exception:
        p50 = 0.0
    try:
        if len(lat) >= 20:
            p95 = statistics.quantiles(lat, n=20)[18]
        else:
            p95 = max(lat) if lat else 0.0
    except Exception:
        p95 = 0.0

    return {
        "count": len(sanitized),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "avg_ms": round(sum(lat)/len(lat), 1) if lat else 0.0,
        "avg_tokens_in": round(sum(tokens_in)/len(tokens_in), 1) if tokens_in else 0.0,
        "avg_tokens_out": round(sum(tokens_out)/len(tokens_out), 1) if tokens_out else 0.0,
        "avg_rougeL": round(statistics.mean(rouge), 3) if rouge else None,
        "avg_bleu": round(statistics.mean(bleu), 3) if bleu else None,
        "avg_evidence": round(statistics.mean(evd), 3) if evd else None,
        "provider_avg_ms": prov_lat,
    }

def get_bench_results() -> Dict[str, Any]:
    """summary + 전체 결과 (항상 기대 스키마 보장)"""
    results = [_sanitize_entry(r) for r in reversed(_bench_cache)]
    return {"summary": get_bench_summary(), "results": results}
