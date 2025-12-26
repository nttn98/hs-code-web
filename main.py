import json
import re
import os
import requests
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template, request, jsonify
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

from customs_advisor_chat import customs_advisor_chat

app = Flask(__name__)

# --- CONFIG ---
load_dotenv()
JSON_PATH = os.getenv("JSON_PATH")  # path tới file output.json
MODEL_AI = os.getenv("MODEL_AI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# --- TOKENIZE ---
def get_tokens(text):
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\wàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ ]"," ",text)
    return set(re.findall(r"\b[a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ0-9]{2,}\b", text))

# --- LOAD JSON DB ---
def load_hs_database(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        row["tokens"] = get_tokens(" ".join([row.get("ten_hang",""), row.get("hs_code","")]))
    return data

# --- SEARCH CANDIDATES ---
def search_candidates(db, query, limit=200):
    q_tokens = get_tokens(query)
    if not q_tokens:
        return []
    scored = []
    for row in db:
        tokens = row.get("tokens", set())
        common = q_tokens & tokens
        if not common:
            continue
        score = len(common)*15 + int(len(common)/max(len(q_tokens),1)*50)
        if q_tokens.issubset(tokens):
            score += 120
        scored.append((score,row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]

# --- PICK HS TOP 1 ---
def pick_hs_code(query, candidates):
    if candidates:
        return {"hs": candidates[0]["hs_code"], "reason": "Top 1 candidate fallback"}
    return None

# --- AI PICK HS CODE ---
def ask_ai_for_hs_code(query, candidates):
    ctx = [{"hs_code": r.get("hs_code"), "ten_hang": r.get("ten_hang"), "chinh_sach": r.get("chinh_sach","")} 
           for r in candidates]
    prompt = f"""
Mô tả hàng hóa: {query}

Danh sách mã HS tham khảo:
{json.dumps(ctx[:50], ensure_ascii=False)}

Yêu cầu:
- Chọn 1 mã HS 8 số phù hợp nhất
- Dựa trên bản chất vật lý và công dụng chính
- Chỉ trả về JSON dạng: {{"hs": "....", "reason": "..."}}
- Nếu không tìm được HS, chọn candidate top 1
"""
    try:
        res = client.chat.completions.create(
            model=MODEL_AI,
            messages=[{"role":"user","content":prompt}]
        )
        content = res.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            data["hs"] = re.sub(r"\D","",data.get("hs",""))
            if not data["hs"] and candidates:
                data["hs"] = candidates[0]["hs_code"]
                data["reason"] = "Fallback top 1"
            return data
        except:
            return {"hs": candidates[0]["hs_code"] if candidates else "N/A", "reason": content}
    except Exception as e:
        return {"hs": "N/A", "reason": str(e)}

# --- CASELAW ---
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

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Thiếu mô tả hàng hóa"}), 400

    if not is_valid_query(query, db):
        return jsonify({"error": "Mô tả hàng hóa không hợp lệ hoặc không tìm thấy kết quả"}), 400

    candidates = search_candidates(db, query)
    hs_res = pick_hs_code(query, candidates)

    if not hs_res:
        hs_res = ask_ai_for_hs_code(query, db)

    hs_final = hs_res.get("hs")
    policy = next((c.get("chinh_sach","") for c in candidates if c.get("hs_code")==hs_final), "")

    return jsonify({
        "hs_code": hs_final,
        "reason": hs_res.get("reason"),
        "policy": policy,
        "caselaw": fetch_caselaw_hierarchy(hs_final)
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message")
    return jsonify({"answer": customs_advisor_chat(msg)})

if __name__ == "__main__":
    app.run(debug=True)
