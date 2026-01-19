import json
import os
import logging
import re
from functools import lru_cache

from flask import Blueprint, request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
from groq import Groq

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# ENV
# =====================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_AI = os.getenv("MODEL_AI", "llama-3.3-70b-versatile")
JSON_GLOBAL = os.getenv("JSON_GLOBAL", "./data/global.json")

client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# BLUEPRINT
# =====================================================
hs_external_bp = Blueprint(
    "hs_external",
    __name__,
    url_prefix="/api/hs/external"
)

# =====================================================
# UTILS
# =====================================================
def extract_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def stream_ai(prompt: str):
    stream = client.chat.completions.create(
        model=MODEL_AI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# =====================================================
# SEMANTIC MAP (VN)
# =====================================================
SEMANTIC_MAP = {
    "heo": ["lợn"],
    "lợn": ["heo"],
    "bò": ["gia súc"],
    "trâu": ["gia súc"],
    "dê": ["gia súc"],
    "cừu": ["gia súc"],
    "gà": ["gia cầm"],
    "vịt": ["gia cầm"],
    "ngan": ["gia cầm"],
    "cá": ["thủy sản"],
    "tôm": ["thủy sản"],
    "mực": ["thủy sản"],
}

def expand_keywords(keywords):
    expanded = set()

    for k in keywords:
        kl = k.lower()
        expanded.add(kl)
        for syn in SEMANTIC_MAP.get(kl, []):
            expanded.add(syn.lower())

    return list(expanded)

# =====================================================
# LOAD DATA
# =====================================================
@lru_cache(maxsize=1)
def load_global_data():
    with open(JSON_GLOBAL, "r", encoding="utf-8") as f:
        return json.load(f)["data"]

# =====================================================
# FLATTEN LEAF (4 DIGITS)
# =====================================================
def flatten_leafs():
    result = []
    for chapter in load_global_data():
        for leaf in chapter.get("leaf", []):
            result.append({
                "leaf_code": leaf.get("code"),
                "leaf_name": leaf.get("name"),
                "groups": leaf.get("group", [])
            })
    return result

FLAT_LEAFS = flatten_leafs()

# =====================================================
# FIND LEAF
# =====================================================
def find_leaf_by_code(leaf_code: str):
    return next(
        (l for l in FLAT_LEAFS if l["leaf_code"] == leaf_code),
        None
    )

# =====================================================
# HS 6 / 8 DIGITS
# =====================================================
def get_hs_codes_by_leaf(leaf):
    items = []
    idx = 1

    for g in leaf.get("groups", []):
        if not g.get("children"):
            if g.get("code") and g.get("name"):
                items.append({
                    "id": idx,
                    "code": g["code"],
                    "name": g["name"]
                })
                idx += 1
            continue

        for c in g.get("children", []):
            if c.get("code") and c.get("name"):
                items.append({
                    "id": idx,
                    "code": c["code"],
                    "name": c["name"]
                })
                idx += 1

    return items

# =====================================================
# AI STEP 1: EXTRACT KEYWORDS
# =====================================================
KEYWORD_PROMPT = """
Trích 3–6 từ khóa quan trọng nhất từ mô tả hàng hóa.
Chỉ trả JSON.

MÔ TẢ:
"{query}"

JSON:
{{
  "keywords": ["...", "..."]
}}
"""

def ai_extract_keywords(query: str):
    buffer = ""
    for t in stream_ai(KEYWORD_PROMPT.format(query=query)):
        buffer += t

    raw = extract_json(buffer)
    if not raw:
        return []

    try:
        return json.loads(raw).get("keywords", [])
    except Exception:
        return []

# =====================================================
# LOCAL FILTER LEAF
# =====================================================
def filter_leaf_by_keywords(keywords, limit=15):
    scored = []

    for leaf in FLAT_LEAFS:
        text = (leaf["leaf_name"] or "").lower()

        for g in leaf.get("groups", []):
            text += " " + (g.get("name") or "").lower()
            for c in g.get("children", []):
                text += " " + (c.get("name") or "").lower()

        score = sum(1 for k in keywords if k in text)
        if score > 0:
            scored.append((score, leaf))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:limit]]

# =====================================================
# AI STEP 2: RANK LEAF LIST
# =====================================================
RANK_LEAF_PROMPT = """
Bạn là chuyên gia HS Code.

Nhiệm vụ:
- Sắp xếp các nhóm HS 4 số theo độ phù hợp (giảm dần)
- KHÔNG tạo mã mới

DANH SÁCH:
{leafs}

MÔ TẢ:
"{query}"

TRẢ JSON:
{{
  "ranking": ["0101", "0103", "..."]
}}
"""

def ai_rank_leafs(query, leafs):
    leaf_text = "\n".join(
        f"{l['leaf_code']} – {l['leaf_name']}"
        for l in leafs
    )

    buffer = ""
    for t in stream_ai(RANK_LEAF_PROMPT.format(
        leafs=leaf_text,
        query=query
    )):
        buffer += t

    raw = extract_json(buffer)
    if not raw:
        return []

    try:
        return json.loads(raw).get("ranking", [])
    except Exception:
        return []

# =====================================================
# API 1: SEARCH LEAF (RETURN LIST)
# =====================================================
@hs_external_bp.route("/search-leaf", methods=["POST"])
def api_search_leaf():
    query = request.json.get("query", "").strip()
    if len(query) < 3:
        return jsonify({"error": "Query quá ngắn"}), 400

    # 1. extract keyword
    keywords = ai_extract_keywords(query)

    # 2. semantic expand
    keywords = expand_keywords(keywords)
    logger.warning("KEYWORDS NORMALIZED: %s", keywords)

    # 3. local filter
    shortlist = filter_leaf_by_keywords(keywords)
    if not shortlist:
        return jsonify({"groups": []})

    # 4. AI ranking
    ranked_codes = ai_rank_leafs(query, shortlist)

    # fallback
    if not ranked_codes:
        return jsonify({
            "groups": [
                {"code": l["leaf_code"], "name": l["leaf_name"]}
                for l in shortlist[:5]
            ]
        })

    result = []
    for code in ranked_codes:
        leaf = find_leaf_by_code(code)
        if leaf:
            result.append({
                "code": leaf["leaf_code"],
                "name": leaf["leaf_name"]
            })

    return jsonify({"groups": result})

# =====================================================
# API 2: FIND BY LEAF
# =====================================================
@hs_external_bp.route("/find-by-leaf", methods=["POST"])
def api_find_by_leaf():
    leaf_code = request.json.get("leaf_code")
    leaf = find_leaf_by_code(leaf_code)

    if not leaf:
        return jsonify({"error": "Leaf không tồn tại"}), 404

    return jsonify({
        "leaf_code": leaf_code,
        "leaf_name": leaf["leaf_name"],
        "items": get_hs_codes_by_leaf(leaf)
    })

# =====================================================
# API 3: CONFIRM HS
# =====================================================
EXACT_PROMPT = """
Bạn là hệ thống CHỌN mã HS từ danh sách CỐ ĐỊNH.

⚠️ QUY TẮC:
- KHÔNG tạo mã mới
- KHÔNG sửa mã
- CHỈ chọn 1 ID
- Không phù hợp → id = -1

NHÓM 4 SỐ:
{leaf_code} – {leaf_name}

DANH SÁCH HS:
{codes}

MÔ TẢ:
"{query}"

TRẢ JSON:
{{
  "id": <number>,
  "reason": "<ngắn>"
}}
"""

@hs_external_bp.route("/confirm-hs", methods=["POST"])
def api_confirm_hs():
    data = request.json
    leaf_code = data.get("leaf_code")
    query = data.get("query", "").strip()

    leaf = find_leaf_by_code(leaf_code)
    if not leaf:
        return jsonify({"error": "Leaf không tồn tại"}), 404

    items = get_hs_codes_by_leaf(leaf)
    id_map = {i["id"]: i for i in items}

    code_lines = [
        f"{i['id']}. {i['code']} – {i['name']}"
        for i in items
    ]

    prompt = EXACT_PROMPT.format(
        leaf_code=leaf_code,
        leaf_name=leaf["leaf_name"],
        codes="\n".join(code_lines),
        query=query
    )

    def generate():
        buffer = ""
        for t in stream_ai(prompt):
            buffer += t

        raw = extract_json(buffer)
        if not raw:
            yield "❌ AI không trả JSON hợp lệ\n\n"
            return

        try:
            result = json.loads(raw)
            selected_id = result.get("id")

            if selected_id not in id_map:
                yield "❌ Không xác định được mã HS phù hợp\n\n"
                return

            item = id_map[selected_id]
            yield f"Mã HS: {item['code']}\n\n"
            yield f"Mô tả: {item['name']}\n\n"

        except Exception as e:
            logger.exception(e)
            yield "❌ Lỗi xử lý AI\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
