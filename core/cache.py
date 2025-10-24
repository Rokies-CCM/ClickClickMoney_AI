# core/cache.py
from __future__ import annotations

import hashlib
import json
import os
import time
import asyncio
from typing import Any, Optional, Dict, Tuple, List

from server.deps import get_redis
from core.utils import simhash, simhash_similarity, normalize_question

DEFAULT_TTL = 60 * 5  # 5분

def _key_hash(prefix: str, payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"

# --- In-memory fallback (thread/async-safe enough for single-process dev) ---
class _InMemoryStore:
    def __init__(self):
        self._data: Dict[str, Tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            now = time.time()
            v = self._data.get(key)
            if not v:
                return None
            exp, payload = v
            if now > exp:
                self._data.pop(key, None)
                return None
            return payload

    async def setex(self, key: str, ttl: int, value: str):
        async with self._lock:
            self._data[key] = (time.time() + ttl, value)

class ResponseCache:
    """정확히 동일한 payload 해시를 키로 사용하는 응답 캐시 (Redis 우선, 메모리 폴백)."""
    def __init__(self, prefix="resp"):
        self.prefix = prefix
        try:
            self.redis = get_redis()  # may raise if not configured
            self._use_memory = False
        except Exception:
            self.redis = _InMemoryStore()
            self._use_memory = True

    async def get(self, payload: dict) -> Optional[dict]:
        key = _key_hash(self.prefix, payload)
        try:
            data = await self.redis.get(key)
        except Exception:
            if not isinstance(self.redis, _InMemoryStore):
                self.redis = _InMemoryStore()
            data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set(self, payload: dict, value: dict, ttl: int = DEFAULT_TTL):
        key = _key_hash(self.prefix, payload)
        body = json.dumps(value, ensure_ascii=False)
        try:
            await self.redis.setex(key, ttl, body)
        except Exception:
            if not isinstance(self.redis, _InMemoryStore):
                self.redis = _InMemoryStore()
            await self.redis.setex(key, ttl, body)

response_cache = ResponseCache()

# -----------------------------------------------------------------------
# Semantic Cache (in-memory only; lightweight and fast) — facts 서명 포함
# -----------------------------------------------------------------------
_SIM_ENABLED = os.getenv("SIMCACHE_ENABLED", "true").strip().lower() in {"1","true","yes","on"}
_SIM_THRESHOLD = float(os.getenv("SIMCACHE_THRESHOLD", "0.90"))
_SIM_CAPACITY = int(os.getenv("SIMCACHE_CAPACITY", "256"))

def _facts_sig(facts: Optional[Dict[str, Any]]) -> str:
    """facts 딕셔너리에 대한 짧은 서명(sha1 8자). facts 없으면 빈 문자열."""
    if not facts:
        return ""
    try:
        payload = json.dumps(facts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return ""

class SemanticCache:
    """
    매우 간단한 의미 캐시 (싱글 프로세스 전제):
    - 키: SimHash(정규화된 질문 + ' [FACTS#<sig>]' 접미) — facts가 다르면 캐시 분리
    - 유사도 >= threshold 이면 히트로 간주
    - LRU-ish + 경년감쇠
    """
    def __init__(self, capacity: int = _SIM_CAPACITY, threshold: float = _SIM_THRESHOLD):
        self.capacity = max(8, int(capacity))
        self.threshold = max(0.5, min(0.99, float(threshold)))
        # (simhash_key, ts, value, facts_sig)
        self._items: List[Tuple[int, float, dict, str]] = []
        self._lock = asyncio.Lock()

    def _make_key(self, question: str, facts: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
        qnorm = normalize_question(question or "")
        sig = _facts_sig(facts)
        base = qnorm if not sig else f"{qnorm} [FACTS#{sig}]"
        return simhash(base), sig

    async def get(self, question: str, facts: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        if not _SIM_ENABLED:
            return None
        key, sig = self._make_key(question, facts)
        if key == 0:
            return None

        best_val, best_score = None, 0.0
        now = time.time()
        async with self._lock:
            for h, ts, val, fsig in self._items:
                # facts 서명이 다르면 다른 문제로 간주 (빈 문자열은 빈 문자열끼리만 허용)
                if sig != fsig:
                    continue
                sim = simhash_similarity(key, h)
                # 10분 초과 시 소폭 감쇠
                age_penalty = 0.0 if (now - ts) < 600 else 0.05
                score = max(0.0, sim - age_penalty)
                if score >= self.threshold and score > best_score:
                    best_score = score
                    best_val = val
                    if score >= 0.99:
                        break
        return best_val

    async def set(self, question: str, value: dict, facts: Optional[Dict[str, Any]] = None):
        if not _SIM_ENABLED:
            return
        key, sig = self._make_key(question, facts)
        if key == 0:
            return
        async with self._lock:
            now = time.time()
            self._items.insert(0, (key, now, value, sig))
            # 같은 simhash 키는 최신 1개만 유지
            seen: set[int] = set()
            dedup: List[Tuple[int, float, dict, str]] = []
            for item in self._items:
                if item[0] in seen:
                    continue
                seen.add(item[0])
                dedup.append(item)
            self._items = dedup[: self.capacity]

semantic_cache = SemanticCache()
