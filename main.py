import json
import re
import os
import unicodedata
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template, request, jsonify
from difflib import SequenceMatcher
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from customs_advisor_chat import customs_advisor_chat

# ================= APP =================
app = Flask(__name__)

# ================= CONFIG =================
load_dotenv()
JSON_PATH = os.getenv("JSON_PATH")          # output.json
JSON_GLOBAL = os.getenv("JSON_GLOBAL")      # global.json
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
    return {t for t in normalize(text).split() if len(t) > 1 and t not in STOPWORDS}

# ================= LOAD OUTPUT.JSON =================
def load_output_db(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        r["_norm"] = normalize(r.get("ten_hang", ""))
        r["_tokens"] = get_tokens(r.get("ten_hang", ""))
    return data

# ================= LOAD GLOBAL.JSON =================
def load_global_db(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["roots"]

output_db = load_output_db(JSON_PATH)
global_roots = load_global_db(JSON_GLOBAL)

# ================= SEARCH OUTPUT.JSON (ABSOLUTE) =================
DANGEROUS_SHORT_TOKENS = {
    "ca", "bo", "ga", "heo", "lon", "vit", "so", "ong", "den"
}

TECH_HINTS = {
    "model", "kw", "hp", "v",
    "dien", "điện",
    "dong co", "động cơ",
    "cong suat", "công suất",
    "dien ap", "điện áp"
}

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def search_output_db(db, query):
    q_norm = normalize(query)
    if not q_norm:
        return None

    q_tokens = get_tokens(query)
    if not q_tokens:
        return None

    allow_fuzzy = len(q_tokens) >= 2
    query_has_tech = any(k in q_norm for k in TECH_HINTS)

    candidates = []

    for r in db:
        r_norm = r["_norm"]
        r_tokens = r["_tokens"]

        score = 0

        # 1️⃣ PHRASE MATCH
        if r_norm == q_norm:
            score += 500
        elif q_norm in r_norm:
            score += 300
        else:
            for n in range(len(q_tokens), 1, -1):
                q_grams = [" ".join(list(q_tokens)[i:i+n]) for i in range(len(q_tokens)-n+1)]
                for g in q_grams:
                    if g in r_norm:
                        score += 150 + n * 10

        # 2️⃣ TOKEN / FUZZY MATCH
        match_tokens = 0
        for qt in q_tokens:
            if qt in DANGEROUS_SHORT_TOKENS:
                continue
            for rt in r_tokens:
                if qt == rt or (allow_fuzzy and similar(qt, rt) >= 0.82):
                    match_tokens += 1
                    break
        score += int(match_tokens / max(len(q_tokens), 1) * 120)

        # 3️⃣ SPECIFICITY RULE
        if query_has_tech:
            if any(k in r_norm for k in TECH_HINTS):
                score += 100
        else:
            if any(k in r_norm for k in TECH_HINTS):
                score -= 120

        # 4️⃣ LENGTH
        score += min(len(r_norm), 200) // 25

        if score > 0:
            candidates.append((score, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    # ===== BẮT BUỘC SCORE THẤP NHẤT ĐỂ CHỌN OUTPUT.JSON =====
    best_score, best_record = candidates[0]
    if best_score < 150:   # score < 150 → không chắc chắn, qua global.json
        return None

    return best_record

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

# ================= AI DOMAIN CLASSIFIER (VERY SMALL PROMPT) =================
def ai_detect_domain(query: str) -> str:
    prompt = f"""
Mô tả: "{query}"

Chọn 1 domain:
animal_food | plant_food | machinery | machinery_part | material | chemical | other

Trả JSON:
{{"domain":"..."}}
"""
    try:
        res = client.chat.completions.create(
            model=MODEL_AI,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30
        )
        return json.loads(res.choices[0].message.content).get("domain", "other")
    except Exception:
        return "other"

# ================= DOMAIN RULE FILTER (NO HS CREATION) =================
def domain_filter(leaves, domain):
    filtered = []
    for l in leaves:
        text = normalize(l["name"] + " " + l["description"])

        if domain == "animal_food":
            if not any(k in text for k in ["ca", "thuy san", "dong vat"]):
                continue

        elif domain == "plant_food":
            if not any(k in text for k in ["thuc vat", "ngu coc", "rau", "qua"]):
                continue

        elif domain in ("machinery", "machinery_part"):
            if not any(k in text for k in ["may", "thiet bi", "phu tung"]):
                continue

        # material / chemical / other → không chặn

        filtered.append(l)

    return filtered

# ================= RANK LEAVES (ANTI 413) =================
def rank_leaves(leaves, query, limit=8):
    q_tokens = get_tokens(query)
    scored = []

    for l in leaves:
        tokens = get_tokens(l["name"] + " " + l["description"])
        score = len(q_tokens & tokens)
        if score > 0:
            scored.append((score, l))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:limit]]

# ================= AI PICKERS (LIMITED CONTEXT) =================
def safe_json(text):
    try:
        return json.loads(text)
    except Exception:
        return {}

def ai_pick_leaf(query, leaves):
    ctx = [{"code": l["code"], "name": l["name"]} for l in leaves]

    prompt = f"""
Mô tả: "{query}"

Danh sách HS:
{json.dumps(ctx, ensure_ascii=False)}

Chọn 1 code phù hợp nhất (không sinh mới).
Trả JSON:
{{"code":"..."}}
"""
    res = client.chat.completions.create(
        model=MODEL_AI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60
    )
    return safe_json(res.choices[0].message.content).get("code")

def ai_pick_group(query, groups):
    ctx = [{"name": g["name"]} for g in groups]

    prompt = f"""
Mô tả: "{query}"

Danh sách phân nhóm:
{json.dumps(ctx, ensure_ascii=False)}

Chọn 1.
Trả JSON:
{{"name":"..."}}
"""
    res = client.chat.completions.create(
        model=MODEL_AI,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60
    )
    return normalize(safe_json(res.choices[0].message.content).get("name", ""))

def pick_final_hs(group):
    children = group.get("children", [])
    if not children:
        return group.get("code")

    for c in children:
        if "loai khac" in normalize(c["name"]):
            return c["code"]

    return children[-1]["code"]

# ================= FETCH CASELAW HIERARCHY =================
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
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = (request.get_json(force=True).get("query") or "").strip()

    # ===== STEP 1: OUTPUT.JSON (ABSOLUTE) =====
    res = search_output_db(output_db, query)
    if res:
        hs_code = res["hs_code"]
        source = "output.json"
        ten_hang = res["ten_hang"]
    else:
        # ===== STEP 2: GLOBAL.JSON =====
        leaves = collect_leaves(global_roots)

        domain = ai_detect_domain(query)
        leaves = domain_filter(leaves, domain)
        leaves = rank_leaves(leaves, query, limit=8)

        if not leaves:
            return jsonify({"error": "Không xác định được HS phù hợp"}), 404

        leaf_code = ai_pick_leaf(query, leaves)
        leaf = next((l for l in leaves if l["code"] == leaf_code), leaves[0])

        groups = leaf.get("group1", [])
        if not groups:
            hs_code = leaf["code"]
            source = "global.json"
            ten_hang = ""
        else:
            group_name = ai_pick_group(query, groups)
            group = next((g for g in groups if normalize(g["name"]) == group_name), groups[0])
            hs_code = pick_final_hs(group)
            source = "global.json"
            ten_hang = leaf["name"]

    # ===== FETCH CASELAW HIERARCHY =====
    caselaw = fetch_caselaw_hierarchy(hs_code)

    # ===== RESPONSE =====
    response = {
        "source": source,
        "hs_code": hs_code,
        "ten_hang": ten_hang,
        "caselaw": caselaw
    }

    if source == "global.json" and groups:
        response["leaf"] = leaf["code"]
        response["group"] = group["name"]

    return jsonify(response)

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
