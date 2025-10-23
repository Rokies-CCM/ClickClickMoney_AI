from typing import List, Dict

def clip_passages(passages: List[Dict], max_chars: int = 1800) -> str:
    out, total = [], 0
    for p in passages:
        t = p.get("text", "")
        if total + len(t) > max_chars:
            t = t[: max(0, max_chars - total)]
        out.append(f"- {t}")
        total += len(t)
        if total >= max_chars: break
    return "\n".join(out)
