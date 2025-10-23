import asyncio
import functools
import re
from typing import Callable, Any, Iterable

import tiktoken

# ---------------------------
# Token counting (tiktoken)
# ---------------------------
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    tiktoken 기반 추정. 실패시 단어 개수로 폴백.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len((text or "").split()))

# ---------------------------
# Text normalization / shingles
# ---------------------------
_WS = re.compile(r"\s+")
_PUNCTS = re.compile(r"[^\w가-힣]+")

def normalize_question(q: str) -> str:
    """
    공백/기호 정리 + 소문자화. (한국어/영문 혼합 친화)
    """
    if not q:
        return ""
    s = q.strip()
    s = _WS.sub(" ", s)
    # 기호 제거(한글/영문/숫자/언더스코어만 남김)
    s = _PUNCTS.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    return s

def _word_tokens(s: str) -> list[str]:
    s = normalize_question(s)
    return [w for w in s.split() if w]

def shingles(words: Iterable[str], k: int = 4) -> list[str]:
    """
    k-gram shingles. (기본 4단어)
    """
    ws = list(words)
    if len(ws) < k:
        return [" ".join(ws)] if ws else []
    return [" ".join(ws[i:i+k]) for i in range(0, len(ws) - k + 1)]

# ---------------------------
# SimHash (의미 캐시용)
# ---------------------------
def _hash64(s: str) -> int:
    # 간단한 64bit 해시
    import hashlib
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)

def simhash(text: str, k: int = 4) -> int:
    """
    SimHash 64bit 구현: k-shingles -> 비트 가중 투표.
    """
    toks = _word_tokens(text)
    sh = shingles(toks, k=k)
    if not sh:
        return 0
    vec = [0] * 64
    for g in sh:
        h = _hash64(g)
        for i in range(64):
            vec[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if vec[i] > 0:
            out |= (1 << i)
    return out

def hamming_distance(a: int, b: int) -> int:
    """
    64bit 해밍거리
    """
    x = (a ^ b) & ((1 << 64) - 1)
    # 파퓰레이션 카운트
    return bin(x).count("1")

def simhash_similarity(a: int, b: int) -> float:
    """
    0.0~1.0 범위 유사도. 64bit 기준.
    """
    if a == 0 and b == 0:
        return 1.0
    dist = hamming_distance(a, b)
    return 1.0 - (dist / 64.0)

# ---------------------------
# Async timeout decorator
# ---------------------------
def with_timeout(seconds: float):
    def deco(fn: Callable[..., Any]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
        return wrapper
    return deco
