import json
import re
import os
import unicodedata
from difflib import SequenceMatcher
from flask import Blueprint, request, jsonify, Response
from dotenv import load_dotenv

load_dotenv()
JSON_PATH = os.getenv("JSON_PATH", "./data/output.json")

hs_internal_bp = Blueprint("hs_internal", __name__, url_prefix="/api/hs/internal")

# ================= NORMALIZE =================
def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str) -> list:
    return normalize(text).split()

# ================= LOAD DATA =================
def load_output_db(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        r["_norm"] = normalize(r.get("ten_hang", ""))
        r["_tokens"] = tokenize(r["_norm"])
    return data

output_db = load_output_db(JSON_PATH)

# ================= ANTI-SPAM =================
def is_spam_query(q: str) -> bool:
    if not q or len(q.strip()) < 3:
        return True
    if len(set(q)) <= 2:
        return True
    if not re.search(r"[a-zA-Zà-ỹÀ-Ỹ]", q):
        return True
    return False

# ================= SEARCH =================
def search_similar_products(query: str, limit=10):
    if is_spam_query(query):
        return []

    q_norm = normalize(query)
    q_tokens = tokenize(q_norm)
    if not q_tokens:
        return []

    head = q_tokens[0]
    results = []

    for r in output_db:
        r_tokens = r["_tokens"]
        if not r_tokens or r_tokens[0] != head:
            continue

        intersect = set(q_tokens) & set(r_tokens)
        coverage = len(intersect) / len(q_tokens)

        if coverage < 0.7:
            continue

        score = coverage * 1000
        score += SequenceMatcher(None, q_norm, r["_norm"]).ratio() * 100

        results.append((score, {
            "hs_code": r.get("hs_code"),
            "ten_hang": r.get("ten_hang"),
            "chinh_sach": r.get("chinh_sach", "")
        }))

    results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in results[:limit]]

# ================= CHAT MEMORY =================
CHAT_MEMORY = {}

# ================= ROUTES =================
@hs_internal_bp.route("/similar", methods=["POST"])
def api_hs_internal_similar():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    return jsonify(search_similar_products(query, limit=10))

@hs_internal_bp.route("/chat", methods=["POST"])
def api_hs_internal_chat():
    from services.internal.hs_policy_ai import ask_hs_policy

    data = request.get_json(force=True)
    hs_code = data.get("hs_code")
    question = (data.get("question") or "").strip()

    if not hs_code or not question:
        return jsonify({"error": "Thiếu hs_code hoặc question"}), 400

    history = CHAT_MEMORY.setdefault(hs_code, [])

    def stream():
        for chunk in ask_hs_policy(hs_code, question, history):
            yield chunk

    return Response(stream(), content_type="text/plain")
