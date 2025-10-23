from __future__ import annotations
import os, glob, json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")
BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()
VECTOR_DB = os.getenv("VECTOR_DB", "data/chroma")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

FAISS_TYPE = os.getenv("FAISS_TYPE", "hnsw").lower()  # flat | hnsw
HNSW_M = int(os.getenv("HNSW_M", "64"))
HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "200"))

# 한국어 문장 분리: Kiwi 우선 -> KSS -> 줄단위
def split_sentences_ko(text: str) -> List[str]:
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        sents = [s.text for s in kiwi.split_into_sents(text)]
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        pass
    try:
        import kss
        return [s.strip() for s in kss.split_sentences(text) if s.strip()]
    except Exception:
        pass
    return [t.strip() for t in text.replace("\r", "").split("\n") if t.strip()]

def load_files() -> List[Dict]:
    paths = []
    for ext in ("*.txt","*.md"):
        paths.extend(glob.glob(str(Path(DOCS_DIR)/ext)))
    out=[]
    for p in paths:
        with open(p,"r",encoding="utf-8",errors="ignore") as f:
            out.append({"path":p,"text":f.read()})
    return out

def chunk_text(text: str, chunk: int, overlap: int) -> List[str]:
    sents = split_sentences_ko(text)
    res, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= chunk:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                res.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + " " + s).strip()
    if buf:
        res.append(buf)
    return res

def main():
    Path(VECTOR_DB).mkdir(parents=True, exist_ok=True)
    items = load_files()
    if not items:
        print(f"[build_index] no files in {DOCS_DIR}")
        return
    emb = SentenceTransformer(EMBED_MODEL)

    docs, metas = [], []
    for it in items:
        chunks = chunk_text(it["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chunks:
            docs.append(c)
            metas.append({"source": it["path"], "title": Path(it["path"]).name, "text": c})
    print(f"[build_index] total chunks: {len(docs)}")

    if BACKEND == "faiss":
        import faiss
        vecs = emb.encode(docs, normalize_embeddings=True).astype("float32")
        dim = vecs.shape[1]

        if FAISS_TYPE == "hnsw":
            # HNSW + cosine
            index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
            index.add(vecs)
            out_name = "faiss_hnsw.index"
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            out_name = "faiss.index"

        faiss.write_index(index, str(Path(VECTOR_DB)/out_name))
        with open(Path(VECTOR_DB)/"faiss_meta.json","w",encoding="utf-8") as f:
            json.dump({i: metas[i] for i in range(len(metas))}, f, ensure_ascii=False)
        print(f"[build_index] FAISS ({FAISS_TYPE}) index built -> {out_name}.")
        return

    # Chroma 백엔드
    from chromadb import PersistentClient
    client = PersistentClient(path=VECTOR_DB)
    col = client.get_or_create_collection("docs", metadata={"hnsw:space":"cosine"})
    vecs = emb.encode(docs, normalize_embeddings=True)
    ids = [f"id-{i}" for i in range(len(docs))]
    B=1000
    for i in range(0,len(docs),B):
        col.add(ids=ids[i:i+B], documents=docs[i:i+B], embeddings=vecs[i:i+B].tolist(), metadatas=metas[i:i+B])
    print("[build_index] Chroma collection built.")

if __name__ == "__main__":
    main()
