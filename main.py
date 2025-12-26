import json
import re
import os
import unicodedata
import requests
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template, request, jsonify
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

from customs_advisor_chat import customs_advisor_chat

app = Flask(__name__)

# ================= CONFIG =================
load_dotenv()
JSON_PATH = os.getenv("JSON_PATH")
MODEL_AI = os.getenv("MODEL_AI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ================= TEXT NORMALIZE =================
def remove_accent_vi(text: str) -> str:
    if not text:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def normalize(text: str, remove_accent=False) -> str:
    if not text:
        return ""
    text = text.lower()
    if remove_accent:
        text = remove_accent_vi(text)

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_tokens(text: str, remove_accent=False) -> set:
    return set(normalize(text, remove_accent).split())

# ================= LOAD JSON DB =================
def load_hs_database(path):
    if not path or not os.path.exists(path):
        raise RuntimeError("JSON_PATH không tồn tại")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        name = row.get("ten_hang", "")
        row["name_norm"] = normalize(name)
        row["name_ascii"] = normalize(name, remove_accent=True)
        row["tokens"] = get_tokens(name)
        row["tokens_ascii"] = get_tokens(name, remove_accent=True)

    return data

db = load_hs_database(JSON_PATH)

# ================= SEARCH ENGINE =================
def search_candidates(db, query, limit=50):
    q_norm = normalize(query)
    q_ascii = normalize(query, remove_accent=True)
    q_tokens = get_tokens(query)
    q_tokens_ascii = get_tokens(query, remove_accent=True)

    scored = []

    for row in db:
        score = 0

        # token match có dấu
        score += len(q_tokens & row["tokens"]) * 40

        # token match không dấu
        score += len(q_tokens_ascii & row["tokens_ascii"]) * 30

        # phrase match
        if q_norm and q_norm in row["name_norm"]:
            score += 200
        if q_ascii and q_ascii in row["name_ascii"]:
            score += 250

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]

# ================= AI QUERY EXPANSION =================
def ai_expand_query(query):
    """
    AI chỉ sinh keyword liên quan
    KHÔNG sinh HS
    """
    prompt = f"""
Người dùng nhập: "{query}"

Sinh tối đa 5 cụm từ mô tả tương đương (tiếng Việt)
để tìm trong database hàng hóa.

Chỉ trả JSON:
{{"keywords": ["...","..."]}}
"""
    try:
        res = client.chat.completions.create(
            model=MODEL_AI,
            messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("keywords", [])
    except:
        return []

# ================= PICK HS =================
def pick_hs_code(candidates):
    if not candidates:
        return None
    return {
        "hs": candidates[0]["hs_code"],
        "reason": "Matched by normalized tokens / phrase"
    }

# ================= CASELAW (OPTIONAL) =================
def fetch_caselaw_hierarchy(hs_code):
    if not hs_code:
        return {"chapter": "", "chapter_groups": {}}
    url = f"https://caselaw.vn/ket-qua-tra-cuu-ma-hs?query={quote_plus(str(hs_code))}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        html_text = resp.text
    except:
        return {"chapter": "", "chapter_groups": {}}

    soup = BeautifulSoup(html_text, "html.parser")
    chapters = []
    chapter_groups = {}
    current_ch = ""
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    chapter_re = re.compile(r'^(Chương\s+\d+)\s*-?\s*(.*)$', re.I)
    code_re = re.compile(r'^(\d{4,10})\s*-?\s*(.*)$')

    for i, ln in enumerate(lines):
        cm = chapter_re.match(ln)
        if cm:
            title = cm.group(1).strip()
            desc = cm.group(2).strip()
            if not desc and i+1 < len(lines):
                desc_candidate = lines[i+1].strip()
                if not desc_candidate.lower().startswith("chuong "):
                    desc = desc_candidate
            chapter_title = f"{title} – {desc}" if desc else title
            chapters.append(chapter_title)
            current_ch = chapter_title
            chapter_groups.setdefault(current_ch, [])
            continue
        cm2 = code_re.match(ln)
        if cm2 and current_ch:
            code = cm2.group(1)
            tail = cm2.group(2).strip()
            if not tail and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if not next_line.lower().startswith("chuong "):
                    tail = next_line
            label = f"{code} – {tail}" if tail else code
            chapter_groups[current_ch].append(label)
    return {"chapter": chapters[0] if chapters else "", "chapter_groups": chapter_groups}

def is_valid_query(query, db, min_tokens=1):
    """
    Kiểm tra query có hợp lệ hay không
    - min_tokens: số token trùng với database tối thiểu
    """
    q_tokens = get_tokens(query)
    if not q_tokens or len(query.strip()) < 3:
        return False

    # kiểm tra trùng token với database
    matched = 0
    for row in db:
        tokens = row.get("tokens", set())
        if q_tokens & tokens:
            matched += 1
            break
    return matched >= min_tokens

# --- LOAD DB ---
db = load_hs_database(JSON_PATH)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not is_valid_query(query, db):
        return jsonify({"error": "Mô tả hàng hóa không hợp lệ hoặc không tìm thấy kết quả"}), 400

    # 1️⃣ search trực tiếp
    candidates = search_candidates(db, query)

    # 2️⃣ nếu yếu → AI mở rộng query
    if not candidates or len(candidates) < 3:
        for kw in ai_expand_query(query):
            candidates.extend(search_candidates(db, kw))

    if not candidates:
        return jsonify({"error": "Không tìm thấy HS phù hợp"}), 404

    hs_res = pick_hs_code(candidates)
    hs_code = hs_res["hs"]

    policy = next(
        (c.get("chinh_sach", "") for c in candidates if c["hs_code"] == hs_code),
        ""
    )

    return jsonify({
        "hs_code": hs_code,
        "reason": hs_res["reason"],
        "policy": policy,
        "caselaw": fetch_caselaw_hierarchy(hs_code)
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    return jsonify({
        "answer": customs_advisor_chat(data.get("message"))
    })

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
