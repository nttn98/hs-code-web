import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

JSON_PATH = os.getenv("JSON_PATH", "./data/output.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DB = json.load(f)

HS_INDEX = {r["hs_code"]: r for r in DB if r.get("hs_code")}

# ================= CHUNK POLICY =================
def chunk_policy(text: str):
    parts = re.split(r"\n|-•|\u2022", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]

# ================= RETRIEVE =================
def retrieve_chunks(chunks, question):
    q = question.lower()
    scored = []
    for c in chunks:
        score = sum(1 for w in q.split() if w in c.lower())
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:5]]

# ================= PROMPT =================
def build_prompt(hs_code, ten_hang, chunks, history, question):
    policy_block = "\n".join([f"- {c}" for c in chunks])
    history_block = "\n".join(
        [f"User: {h['q']}\nAI: {h['a']}" for h in history[-3:]]
    )

    return f"""
Bạn là CHUYÊN GIA TƯ VẤN HS CODE cho DOANH NGHIỆP (internal tool).

QUY TẮC BẮT BUỘC:
- CHỈ dùng thông tin bên dưới
- KHÔNG suy đoán
- KHÔNG bổ sung luật
- Không có dữ liệu → nói rõ chưa có thông tin

HS CODE: {hs_code}
TÊN HÀNG: {ten_hang}

TRÍCH DẪN CHÍNH SÁCH:
{policy_block}

LỊCH SỬ TRAO ĐỔI:
{history_block}

YÊU CẦU TRẢ LỜI:
- Văn phong tư vấn pháp lý
- Gạch đầu dòng
- Mỗi ý phải kèm (Trích: "...")

CÂU HỎI:
{question}
"""

# ================= CHAT =================
def ask_hs_policy(hs_code, question, history):
    record = HS_INDEX.get(hs_code)
    if not record:
        yield "❌ HS Code không tồn tại trong dữ liệu."
        return

    policy = (record.get("chinh_sach") or "").strip()
    if not policy:
        yield "⚠️ Chưa có dữ liệu chính sách cho HS Code này."
        return

    chunks = chunk_policy(policy)
    rel = retrieve_chunks(chunks, question)

    if not rel:
        yield "📌 Dữ liệu hiện tại chưa ghi nhận thông tin cho câu hỏi này."
        return

    prompt = build_prompt(
        hs_code,
        record.get("ten_hang", ""),
        rel,
        history,
        question
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_completion_tokens=900,
        stream=True
    )

    answer = ""
    for chunk in completion:
        text = chunk.choices[0].delta.content or ""
        answer += text
        yield text

    history.append({"q": question, "a": answer})
