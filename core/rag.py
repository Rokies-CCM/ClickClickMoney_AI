# core/rag.py
from __future__ import annotations

import json
import logging
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from core.compress import clip_passages
from core.config import get_settings

cfg = get_settings()

# ------------------------------------------------------------------------------
# Logger & warn-once
# ------------------------------------------------------------------------------
log = logging.getLogger("core.rag")
_warned: Set[str] = set()

def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)

# ------------------------------------------------------------------------------
# Embedder (lazy)
# ------------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    """
    Lazy-load SentenceTransformer with name from config.
    """
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
            log.info(f"[RAG] SentenceTransformer loaded: {cfg.EMBEDDING_MODEL}")
        except Exception as e:
            _warn_once("embedder_load_fail", f"Embedder load failed ({cfg.EMBEDDING_MODEL}): {e}")
            raise
    return _embedder

def _encode(texts: List[str]) -> np.ndarray:
    """
    SBERT encode with normalization when available (older versions fallback).
    """
    model = get_embedder()
    try:
        emb = model.encode(texts, normalize_embeddings=True)
    except TypeError:
        emb = model.encode(texts)
    return np.array(emb).astype("float32")

# ------------------------------------------------------------------------------
# Vector stores (lazy)
# ------------------------------------------------------------------------------
_chroma = None
_faiss = None
_idx_meta: Dict[int, Dict] = {}

def _lazy_load_chroma():
    """
    Returns Chroma collection or None.
    """
    global _chroma
    if _chroma is not None:
        return _chroma
    try:
        from chromadb import PersistentClient  # optional dependency at runtime
        path = cfg.VECTOR_DB
        client = PersistentClient(path=path)
        _chroma = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
        log.info(f"[RAG] Chroma collection ready at {path}")
    except Exception as e:
        _chroma = None
        _warn_once("chroma_load_fail", f"Chroma load failed: {e}")
    return _chroma

def _lazy_load_faiss():
    """
    Returns (index, meta). When unavailable, returns (None, {}).
    """
    global _faiss, _idx_meta
    if _faiss is not None:
        return _faiss, _idx_meta
    try:
        import faiss  # optional dependency
        base = cfg.VECTOR_DB
        cand = [("faiss_hnsw.index", "hnsw"), ("faiss.index", "flat")]
        index_path = None
        for name, _ in cand:
            p = f"{base}/{name}"
            try:
                import os
                if os.path.exists(p):
                    index_path = p
                    break
            except Exception:
                pass
        if index_path is None:
            index_path = f"{base}/faiss.index"

        meta_path = f"{base}/faiss_meta.json"
        try:
            import os
            ok = os.path.exists(index_path) and os.path.exists(meta_path)
        except Exception:
            ok = False

        if not ok:
            _faiss = None
            _idx_meta = {}
            _warn_once("faiss_missing", f"FAISS index/meta not found at {base} (index={index_path.split('/')[-1]})")
            return _faiss, _idx_meta

        _faiss = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _idx_meta = {int(k): v for k, v in raw.items()}
        log.info(f"[RAG] FAISS index loaded: {index_path} | meta={len(_idx_meta)}")
    except Exception as e:
        _faiss = None
        _idx_meta = {}
        _warn_once("faiss_load_fail", f"FAISS load failed: {e}")
    return _faiss, _idx_meta

# ------------------------------------------------------------------------------
# Re-ranker (Upstage)
# ------------------------------------------------------------------------------
def rerank_upstage(query: str, passages: List[Dict], top_k: int) -> List[Dict]:
    """
    Upstage Rerank API (sync). Falls back gracefully on any error or missing key.
    Expects passages as [{"text":..., "source":..., "title":...}, ...]
    """
    if not (cfg.USE_RERANK and cfg.UPSTAGE_API_KEY):
        return passages[:top_k]
    import httpx
    url = "https://api.upstage.ai/v1/retrieval/rerank"
    headers = {"Authorization": f"Bearer {cfg.UPSTAGE_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": query, "passages": [p.get("text", "") for p in passages], "top_k": top_k}
    try:
        # keep simple; rely on server-wide HTTP/2 elsewhere
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json() or {}
            scores = data.get("scores") or data.get("results") or []
            # Case A: scores: List[float]
            if isinstance(scores, list) and scores and isinstance(scores[0], (int, float)):
                ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
                return [passages[i] for i, _ in ranked[:top_k]]
            # Case B: results: List[{"index": int, "score": float}]
            if scores and isinstance(scores[0], dict) and "index" in scores[0]:
                order = sorted(scores, key=lambda x: x.get("score", 0.0), reverse=True)
                return [passages[min(it.get("index", 0), len(passages) - 1)] for it in order[:top_k]]
            return passages[:top_k]
    except Exception as e:
        _warn_once("rerank_upstage_fail", f"Upstage rerank failed: {e}")
        return passages[:top_k]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _to_passage(text: str, meta: Dict) -> Dict:
    return {"text": text, "source": meta.get("source", ""), "title": meta.get("title", "")}

def _dynamic_k_and_context(query: str, base_k: int, base_maxc: int) -> Tuple[int, int]:
    qlen = len((query or "").strip())
    if qlen < 40:
        return max(1, min(2, base_k)), min(800, base_maxc)
    if qlen < 120:
        return max(2, min(3, base_k)), min(1200, base_maxc)
    return base_k, base_maxc

# ------------------------------------------------------------------------------
# Search (local)
# ------------------------------------------------------------------------------
def search_local(query: str, k: int) -> List[Dict]:
    """
    Searches local vector store (FAISS or Chroma) and returns passages.
    """
    # FAISS first (fast & light)
    if cfg.VECTOR_BACKEND == "faiss":
        index, idx_meta = _lazy_load_faiss()
        if index is None or not idx_meta:
            # silently fallback to chroma if available
            pass
        else:
            try:
                import faiss
                emb = _encode([query])
                # Set efSearch only when supported (HNSW)
                try:
                    if hasattr(index, "hnsw"):
                        index.hnsw.efSearch = int(cfg.FAISS_EF_SEARCH)
                except Exception as e:
                    _warn_once("faiss_efsearch_fail", f"FAISS efSearch set failed: {e}")
                D, I = index.search(emb, k)
                out: List[Dict] = []
                for idx in I[0]:
                    if idx < 0:
                        continue
                    meta = idx_meta.get(int(idx), {})
                    out.append(_to_passage(meta.get("text", ""), meta))
                if not out:
                    _warn_once("faiss_empty", f"FAISS search returned no results (k={k}, qlen={len(query)})")
                return out
            except Exception as e:
                _warn_once("faiss_search_fail", f"FAISS search failed: {e}")

    # Chroma fallback or selected backend
    try:
        col = _lazy_load_chroma()
        if col is None:
            return []
        res = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = [_to_passage(t, (m or {})) for t, m in zip(docs, metas)]
        if not out:
            _warn_once("chroma_empty", f"Chroma returned no results (k={k}, qlen={len(query)})")
        return out
    except Exception as e:
        _warn_once("chroma_query_fail", f"Chroma query failed: {e}")
        return []

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
def build_context(
    query: str,
    web_snippets: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    *,
    min_k: Optional[int] = None,
    prefer_web_first: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    질의에 대한 컨텍스트 문자열과 사용된 패시지 목록을 구성한다.
    - min_k: 동적 축소(k=1~2) 상황에서도 최소 k를 보장 (뉴스성 짧은 질의용)
    - prefer_web_first: 웹 스니펫을 로컬보다 앞에 배치(슬라이스 시 웹 우선 포함)
    반환:
      ctx: str           -> 모델에게 줄 컨텍스트(길이 제한 clip)
      passages: List[Dict] -> 실제 포함된 패시지(웹+로컬 혼합)
    """
    base_k = top_k if top_k is not None else cfg.TOP_K
    base_maxc = max_context_chars if max_context_chars is not None else cfg.MAX_CONTEXT_CHARS

    k, maxc = _dynamic_k_and_context(query, base_k, base_maxc)
    if min_k is not None and k < min_k:
        k = int(min_k)

    local = search_local(query, k=k)
    passages: List[Dict] = []

    if prefer_web_first:
        if web_snippets:
            passages.extend(web_snippets)
        passages.extend(local)
    else:
        passages.extend(local)
        if web_snippets:
            passages.extend(web_snippets)

    # 긴 질문에서만 재랭크 (웹 우선 유지 시에는 재랭크 건너뜀)
    if len((query or "").strip()) >= 120 and k >= 4 and not prefer_web_first:
        passages = rerank_upstage(query, passages, top_k=k)
    else:
        passages = passages[:k]

    # 컨텍스트 문자열 생성 (길이 제한)
    ctx = clip_passages(passages, max_chars=maxc)

    if not passages:
        _warn_once("context_empty", f"Context empty (backend={cfg.VECTOR_BACKEND}, k={k}, maxc={maxc}, qlen={len(query)})")

    return ctx, passages[:k]
