import os
import re
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong file .env")

# Có thể đổi sang model nhẹ hơn để tiết kiệm token
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")

client = Groq(api_key=GROQ_API_KEY)


def format_hs_code(hs: str) -> str:
    digits = re.sub(r"\D", "", str(hs))
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    elif len(digits) == 6:
        return f"{digits[:4]}.{digits[4:]}"
    return str(hs)


def build_context(candidates: List[Dict]) -> str:
    """
    Gom các dòng HS ứng viên thành chuỗi ngắn gọn, giới hạn mô tả <= 150 ký tự.
    """
    lines = []
    for i, row in enumerate(candidates[:30], start=1):  # chỉ lấy tối đa 30 dòng
        hs_raw = row.get("hs_code", "")
        desc = str(row.get("description", ""))[:150]
        lines.append(f"{i}. {hs_raw}: {desc}")
    return "\n".join(lines)


def _extract_hs(text: str) -> str | None:
    m = re.search(r"\b\d{4}(?:\.\d{2}(?:\.\d{2})?)?\b", text)
    return m.group(0) if m else None


def ask_model_for_hs(query: str, candidate_rows) -> str:
    candidates = candidate_rows.to_dict(orient="records")
    if not candidates:
        return "Không tìm được mã HS ứng viên nào."

    context = build_context(candidates)

    system_prompt = (
        "Bạn là chuyên gia phân loại mã HS Việt Nam. "
        "Hãy chọn 1 mã HS phù hợp nhất từ danh sách, không được tự nghĩ thêm."
    )

    user_prompt = f"""
Mô tả hàng hóa: "{query}"

Danh sách ứng viên:
{context}

Trả về duy nhất 1 dòng:
HS code: <mã HS>
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=16,  # maybe increase to 64 or 256
        )
        output = completion.choices[0].message.content or ""
    except RateLimitError:
        return "🚫 Hết giới hạn Groq trong ngày, vui lòng thử lại sau."

    chosen_hs = _extract_hs(output)
    if not chosen_hs:
        chosen_hs = candidates[0].get("hs_code", "")

    chosen_digits = re.sub(r"\D", "", chosen_hs)
    matched_rows = [
        row for row in candidates
        if re.sub(r"\D", "", str(row.get("hs_code", ""))) == chosen_digits
    ] or [candidates[0]]

    formatted_code = format_hs_code(matched_rows[0].get("hs_code", ""))

    names = []
    for row in matched_rows:
        n = row.get("ten_hang") or row.get("description")
        n = str(n).strip()
        if n and n not in names:
            names.append(n)

    result_lines = [f"HS code: {formatted_code}"]
    if names:
        result_lines.append("Các tên hàng khớp trong dữ liệu:")
        for name in names[:8]:  # 🔹 giới hạn hiển thị 8 tên hàng
            result_lines.append(f"- {name}")

    return "\n".join(result_lines)
