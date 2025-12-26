import json
import re
import os
import unicodedata
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template, request, jsonify

from customs_advisor_chat import customs_advisor_chat

app = Flask(__name__)

# ================= CONFIG =================
load_dotenv()
JSON_PATH = os.getenv("JSON_PATH")            # output.json
JSON_GLOBAL = os.getenv("JSON_GLOBAL")        # global.json
MODEL_AI = os.getenv("MODEL_AI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ================= TEXT NORMALIZE =================
STOPWORDS = {
    "va", "và", "loai", "loại", "dung", "dùng", "de", "để",
    "bang", "bằng", "trong", "cho", "tu", "từ"
}

def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_tokens(text: str) -> set:
    return {
        t for t in normalize(text).split()
        if len(t) > 1 and t not in STOPWORDS
    }

# ================= LOAD OUTPUT.JSON =================
def load_output_db(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        norm = normalize(r.get("ten_hang", ""))
        r["_norm"] = norm
        r["_tokens"] = get_tokens(norm)
    return data

# ================= LOAD GLOBAL.JSON =================
def load_global_db(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["roots"]

output_db = load_output_db(JSON_PATH)
global_roots = load_global_db(JSON_GLOBAL)

# ================= SEARCH OUTPUT.JSON =================
def search_output_db(db, query):
    q_norm = normalize(query)
    q_tokens = get_tokens(query)

    if len(q_tokens) < 2:
        return None

    best = None
    best_score = 0

    for r in db:
        common = q_tokens & r["_tokens"]
        if not common:
            continue

        score = len(common) * 50

        if q_norm in r["_norm"]:
            score += 300

        coverage = len(common) / len(q_tokens)
        score += int(coverage * 100)

        if score > best_score:
            best_score = score
            best = r

    return best if best_score >= 180 else None

# ================= GLOBAL HELPERS =================
def collect_leaves(roots):
    leaves = []
    for r in roots:
        for leaf in r.get("leaf", []):
            leaves.append({
                "code": leaf["code"],
                "name": leaf["name"],
                "description": r.get("description", ""),
                "group1": leaf.get("group1", [])
            })
    return leaves

def filter_leaves_by_state(leaves, query):
    q_tokens = get_tokens(query)
    if not q_tokens:
        return []

    filtered = []
    for l in leaves:
        text = f"{l['name']} {l['description']}"
        tokens = get_tokens(text)
        if q_tokens & tokens:
            filtered.append(l)

    return filtered

# ================= AI HELPERS =================
def safe_ai_json(text):
    try:
        return json.loads(text)
    except Exception:
        return {}

def ai_pick_leaf(query, leaves):
    ctx = [{"code": l["code"], "name": l["name"]} for l in leaves]

    prompt = f"""
Mô tả hàng hóa: "{query}"

Danh sách nhóm HS:
{json.dumps(ctx, ensure_ascii=False)}

Chọn 1 nhóm PHÙ HỢP NHẤT.
KHÔNG sinh mã mới.

Trả JSON:
{{"code": "..."}}
"""
    res = client.chat.completions.create(
        model=MODEL_AI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return safe_ai_json(res.choices[0].message.content).get("code")

def ai_pick_group(query, groups):
    ctx = [{"name": g["name"]} for g in groups]

    prompt = f"""
Mô tả hàng hóa: "{query}"

Danh sách phân nhóm:
{json.dumps(ctx, ensure_ascii=False)}

Chọn 1 phân nhóm PHÙ HỢP NHẤT.
Trả JSON:
{{"name": "..."}}
"""
    res = client.chat.completions.create(
        model=MODEL_AI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return normalize(
        safe_ai_json(res.choices[0].message.content).get("name", "")
    )

def pick_final_hs(group):
    children = group.get("children", [])
    if not children:
        return group.get("code")

    for c in children:
        if "loại khác" in normalize(c["name"]):
            return c["code"]

    return children[-1]["code"]

# ================= CASELAW FETCH (ADD – KHÔNG ĐỔI LOGIC) =================
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

# ================= ROUTES =================
def rank_leaves(leaves, query, limit=12):
    q_tokens = get_tokens(query)
    scored = []

    for l in leaves:
        text = f"{l['name']} {l['description']}"
        tokens = get_tokens(text)
        score = len(q_tokens & tokens)
        if score > 0:
            scored.append((score, l))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:limit]]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = (request.get_json(force=True).get("query") or "").strip()

    # STEP 1: output.json
    res = search_output_db(output_db, query)
    if res:
        caselaw = fetch_caselaw_hierarchy(res["hs_code"])
        return jsonify({
            "source": "output.json",
            "hs_code": res["hs_code"],
            "ten_hang": res["ten_hang"],
            "caselaw": caselaw
        })

    # STEP 2: global.json
    leaves = collect_leaves(global_roots)
    leaves = filter_leaves_by_state(leaves, query)

    if not leaves:
        return jsonify({"error": "Không xác định được nhóm HS"}), 404

    leaves = rank_leaves(leaves, query, limit=10)

    leaf_code = ai_pick_leaf(query, leaves)
    leaf = next((l for l in leaves if l["code"] == leaf_code), leaves[0])

    groups = leaf.get("group1", [])
    if not groups:
        caselaw = fetch_caselaw_hierarchy(leaf["code"])
        return jsonify({
            "source": "global.json",
            "hs_code": leaf["code"],
            "leaf": leaf["code"],
            "caselaw": caselaw
        })

    group_name = ai_pick_group(query, groups)
    group = next(
        (g for g in groups if normalize(g["name"]) == group_name),
        groups[0]
    )

    hs_final = pick_final_hs(group)

    caselaw = fetch_caselaw_hierarchy(hs_final)

    return jsonify({
        "source": "global.json",
        "hs_code": hs_final,
        "leaf": leaf["code"],
        "group": group["name"],
        "caselaw": caselaw
    })

@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({
        "answer": customs_advisor_chat(
            request.get_json(force=True).get("message")
        )
    })

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
