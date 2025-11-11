# model_client.py
"""
Optimized model client for HS code selection using Groq LLM.

Main ideas to save tokens while keeping accuracy:
- Local pre-filter: compute similarity between user query and each candidate.description
  using difflib.SequenceMatcher and keyword overlap, then keep top_k candidates only.
- Cache results (cache.json) to avoid repeated LLM calls for same/similar queries.
- Minimal, deterministic prompt + low max_tokens to reduce output tokens.
- Fallback to best local candidate if LLM fails or rate-limited.
"""
import os
import re
import json
import time
import math
import hashlib
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from difflib import SequenceMatcher

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong file .env")

# Config (can override via env)
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
MAX_TOKENS = int(os.environ.get("GC_MAX_TOKENS", "64"))
TEMPERATURE = float(os.environ.get("GC_TEMPERATURE", "0"))
TOP_K = int(os.environ.get("GC_TOP_K", "5"))           # số ứng viên gửi lên model
CACHE_PATH = os.environ.get("GC_CACHE_PATH", "cache.json")
SIMILARITY_THRESHOLD_FALLBACK = float(os.environ.get("GC_SIM_FALLBACK", "0.72"))
# If top local similarity >= threshold and LLM unavailable, we fallback to that.

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------
# Utility helpers
# ---------------------------
def _norm_text(s: str) -> str:
    """Normalize text for caching / matching (lowercase, strip, collapse spaces)."""
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", str(s).strip().lower())
    return t


def _hash_query(s: str) -> str:
    """Stable hash for query to use in cache keys (use normalized text)."""
    h = hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()
    return h


def format_hs_code(hs: str) -> str:
    digits = re.sub(r"\D", "", str(hs))
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    if len(digits) == 6:
        return f"{digits[:4]}.{digits[4:]}"
    return str(hs)


def _similarity(a: str, b: str) -> float:
    """Return similarity score in [0,1] between a and b using SequenceMatcher."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap_score(query: str, text: str) -> float:
    """Simple keyword overlap: fraction of query words present in text."""
    q_words = [w for w in re.split(r"\W+", query.lower()) if w and len(w) > 1]
    if not q_words:
        return 0.0
    t = text.lower()
    hit = sum(1 for w in q_words if w in t)
    return hit / len(q_words)


# ---------------------------
# Cache functions
# ---------------------------
def _load_cache(path: str = CACHE_PATH) -> Dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict, path: str = CACHE_PATH) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


# ---------------------------
# Pre-filtering: choose top_k candidates locally
# ---------------------------
def rank_candidates_by_relevance(query: str, candidates: List[Dict], top_k: int = TOP_K
                                 ) -> List[Tuple[Dict, float]]:
    """
    Score each candidate by combination of SequenceMatcher similarity and keyword overlap,
    then return top_k (candidate, score) sorted desc.
    """
    q_norm = _norm_text(query)
    scored = []
    for row in candidates:
        desc = _norm_text(row.get("description", "") or "")
        # compute scores
        sim = _similarity(q_norm, desc)
        kw = _keyword_overlap_score(q_norm, desc)
        # combine: weighted sum (favor similarity but boost keyword match)
        score = 0.65 * sim + 0.35 * kw
        scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max(1, min(top_k, len(scored)))]


# ---------------------------
# LLM call (minimal prompt)
# ---------------------------
def _call_model_select_hs(query: str, top_candidates: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Groq LLM with a minimal prompt and the small list of top_candidates.
    Returns (chosen_hs_text, raw_model_output).
    chosen_hs_text is like '6403.99.90' or '64039990' or '6403.99'
    """
    # build compact context: index. hs_raw: shortdesc (truncate)
    ctx_lines = []
    for i, row in enumerate(top_candidates, start=1):
        hs_raw = row.get("hs_code", "")
        desc = str(row.get("description", "") or "")[:140].replace("\n", " ")
        ctx_lines.append(f"{i}. {hs_raw} | {desc}")
    context = "\n".join(ctx_lines)

    system_prompt = (
        "Bạn là chuyên gia phân loại mã HS Việt Nam. "
        "Từ danh sách ngắn dưới đây, CHỈ CHỌN MỘT mã HS có sẵn và TRẢ VỀ DUY NHẤT một dòng theo format: HS code: <mã>"
    )

    user_prompt = f'Mô tả: "{query}"\n\nỨng viên:\n{context}\n\nTrả về duy nhất:\nHS code: <mã HS trong danh sách>'

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        output = completion.choices[0].message.content or ""
        # extract hs
        m = re.search(r"\b\d{4}(?:\.\d{2}(?:\.\d{2})?)?\b", output)
        chosen = m.group(0) if m else None
        return chosen, output
    except RateLimitError:
        # bubble up to caller for fallback
        raise
    except Exception:
        return None, None


# ---------------------------
# Main API
# ---------------------------
def ask_model_for_hs(query: str, candidate_rows) -> str:
    """
    Main function to call from app:
    - candidate_rows: pandas DataFrame (with columns 'hs_code', 'description', optionally 'ten_hang')
    Returns multi-line string:
      HS code: xxxx.xx.xx
      Các tên hàng khớp trong dữ liệu:
      - ...
    """
    # load cache
    cache = _load_cache()
    q_norm = _norm_text(query)
    q_key = _hash_query(q_norm)

    # 1) if cached exact query -> return immediately
    if q_key in cache:
        return cache[q_key]

    # prepare candidates list
    candidates = candidate_rows.to_dict(orient="records")
    if not candidates:
        return "Không tìm được mã HS ứng viên nào."

    # 2) local ranking -> top_k
    ranked = rank_candidates_by_relevance(query, candidates, top_k=TOP_K)
    top_candidates = [r for r, score in ranked]
    top_scores = [score for r, score in ranked]

    # If top candidate is very clearly best (score >> others), we can skip LLM and use it directly.
    best_score = top_scores[0] if top_scores else 0.0
    second_score = top_scores[1] if len(top_scores) > 1 else 0.0

    # Heuristic: if best is much better than second OR absolute threshold high, fallback to local
    if best_score >= 0.92 or (best_score - second_score) >= 0.20 and best_score >= 0.75:
        chosen_row = top_candidates[0]
        chosen_digits = re.sub(r"\D", "", str(chosen_row.get("hs_code", "")))
        # build result
        formatted_code = format_hs_code(chosen_row.get("hs_code", ""))
        # names list
        names = []
        for row in candidates:
            if re.sub(r"\D", "", str(row.get("hs_code", ""))) == chosen_digits:
                n = row.get("ten_hang") or row.get("description")
                n = str(n).strip()
                if n and n not in names:
                    names.append(n)
        lines = [f"HS code: {formatted_code}"]
        if names:
            lines.append("Các tên hàng khớp trong dữ liệu:")
            for name in names[:8]:
                lines.append(f"- {name}")
        result_text = "\n".join(lines)
        # cache and return
        cache[q_key] = result_text
        _save_cache(cache)
        return result_text

    # 3) Otherwise call LLM with top_candidates only (reduces input tokens a lot)
    try:
        chosen_hs_text, model_output = _call_model_select_hs(query, top_candidates)
    except RateLimitError:
        # fallback: use best local candidate if rate limited
        chosen_row = top_candidates[0]
        chosen_digits = re.sub(r"\D", "", str(chosen_row.get("hs_code", "")))
        formatted_code = format_hs_code(chosen_row.get("hs_code", ""))
        names = []
        for row in candidates:
            if re.sub(r"\D", "", str(row.get("hs_code", ""))) == chosen_digits:
                n = row.get("ten_hang") or row.get("description")
                n = str(n).strip()
                if n and n not in names:
                    names.append(n)
        lines = [f"HS code: {formatted_code}"]
        if names:
            lines.append("Các tên hàng khớp trong dữ liệu:")
            for name in names[:8]:
                lines.append(f"- {name}")
        result_text = "\n".join(lines)
        cache[q_key] = result_text
        _save_cache(cache)
        return result_text

    # if model didn't return a valid hs, fallback to best local candidate
    if not chosen_hs_text:
        chosen_row = top_candidates[0]
    else:
        # find matching rows by digits
        chosen_digits = re.sub(r"\D", "", chosen_hs_text)
        matched_rows = [
            row for row in candidates
            if re.sub(r"\D", "", str(row.get("hs_code", ""))) == chosen_digits
        ]
        if matched_rows:
            chosen_row = matched_rows[0]
        else:
            # model chose something not found (shouldn't happen) -> fallback
            chosen_row = top_candidates[0]

    # Build final result (same format)
    chosen_digits = re.sub(r"\D", "", str(chosen_row.get("hs_code", "")))
    formatted_code = format_hs_code(chosen_row.get("hs_code", ""))
    names = []
    for row in candidates:
        if re.sub(r"\D", "", str(row.get("hs_code", ""))) == chosen_digits:
            n = row.get("ten_hang") or row.get("description")
            n = str(n).strip()
            if n and n not in names:
                names.append(n)

    result_lines = [f"HS code: {formatted_code}"]
    if names:
        result_lines.append("Các tên hàng khớp trong dữ liệu:")
        for name in names[:8]:
            result_lines.append(f"- {name}")

    result_text = "\n".join(result_lines)

    # Cache result
    cache[q_key] = result_text
    # keep cache size bounded (optional): keep latest 2000 entries
    try:
        if len(cache) > 2000:
            # naive trim by keys order isn't LRU but ok for small apps
            keys = list(cache.keys())[-2000:]
            new_cache = {k: cache[k] for k in keys}
            cache = new_cache
    except Exception:
        pass

    _save_cache(cache)
    return result_text
