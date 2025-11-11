import json
import re
from pathlib import Path

import requests


# ================== CONFIG ==================
JSON_DB_FILE = "hs_codes.json"          # file dữ liệu HS đã export
OLLAMA_URL = "http://localhost:11434"   # endpoint ollama
OLLAMA_MODEL = "llama3.2:3b"                # ví dụ: model hỗ trợ tiếng Việt tốt
TOP_K_CANDIDATES = 30                   # số dòng gửi cho model chọn


# =============== UTIL ===============
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_hs_data(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("File JSON phải là list các object.")
    return data


def score_record(query_norm: str, rec) -> int:
    """
    Tính điểm đơn giản: số từ trùng giữa query và text mô tả trong record.
    Dùng cho bước lọc sơ bộ (không dùng để thay HS code).
    """
    text = " ".join([
        rec.get("ten_hang", ""),
        rec.get("ghi_chu_1", ""),
        rec.get("ghi_chu_2", ""),
    ])

    text_norm = normalize_text(text)
    if not text_norm:
        return 0

    q_words = set(query_norm.split())
    t_words = set(text_norm.split())
    return len(q_words & t_words)


def select_candidates(query: str, db: list, top_k: int = TOP_K_CANDIDATES):
    query_norm = normalize_text(query)
    scored = []
    for rec in db:
        score = score_record(query_norm, rec)
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Nếu tất cả score = 0: vẫn trả top_k đầu cho model chọn
    if scored and scored[0][0] == 0:
        return [rec for _, rec in scored[:top_k]]

    positive = [rec for s, rec in scored if s > 0]
    if not positive:
        return [rec for _, rec in scored[:top_k]]
    return positive[:top_k]


def build_prompt(user_query: str, candidates: list) -> str:
    """
    Tạo prompt cho Ollama.
    RẤT QUAN TRỌNG: model chỉ được chọn index (1,2,3,...) chứ không được tự tạo HS code.
    """

    lines = []
    for i, rec in enumerate(candidates, start=1):
        hs = rec.get("hs_code", "")
        ten = rec.get("ten_hang", "")
        g1 = rec.get("ghi_chu_1", "")
        g2 = rec.get("ghi_chu_2", "")
        item_line = f"[{i}] HS: {hs} | Tên hàng: {ten}"
        if g1:
            item_line += f" | Ghi chú 1: {g1}"
        if g2:
            item_line += f" | Ghi chú 2: {g2}"
        lines.append(item_line)

    db_text = "\n".join(lines)

    prompt = f"""
Bạn là chuyên gia phân loại HS code trong lĩnh vực xuất nhập khẩu.

Dưới đây là một DANH SÁCH CỐ ĐỊNH các mặt hàng và mã HS tương ứng (DATABASE).
BẠN KHÔNG ĐƯỢC TỰ TẠO HS CODE MỚI, chỉ được chọn trong danh sách này.

DATABASE:
{db_text}

Người dùng sẽ nhập mô tả hàng hóa bằng TIẾNG VIỆT hoặc TIẾNG ANH.

MÔ TẢ CỦA NGƯỜI DÙNG:
\"\"\"{user_query}\"\"\"

YÊU CẦU:
1. Phân tích mô tả hàng hóa của người dùng.
2. So khớp với danh sách DATABASE ở trên.
3. Chọn ra MỘT dòng phù hợp nhất bằng cách chọn SỐ THỨ TỰ [i] trong danh sách.
4. KHÔNG được tạo HS CODE mới, KHÔNG được sửa HS CODE. Chỉ dùng đúng HS trong DATABASE.
5. Trả lời duy nhất ở dạng JSON với cấu trúc:

{{
  "index_in_list": <số thứ tự [i] trong DATABASE>,
  "explanation_vi": "<giải thích ngắn gọn bằng tiếng Việt>",
  "explanation_en": "<short explanation in English>"
}}

Lưu ý:
- "index_in_list" là số nguyên, từ 1 đến {len(candidates)}.
- KHÔNG trả về thêm bất kỳ text nào ngoài JSON.
"""
    return prompt.strip()


def call_ollama(prompt: str) -> str:
    """
    Gọi Ollama /api/generate với stream=False.
    In ra thông tin lỗi nếu server trả về 4xx/5xx.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload)
    except requests.exceptions.ConnectionError as e:
        print("❌ Không kết nối được tới Ollama. Bạn đã chạy ollama chưa?")
        print(f"Chi tiết lỗi: {e}")
        raise

    if not resp.ok:
        print("❌ Ollama trả về lỗi HTTP:")
        print(f"Status code: {resp.status_code}")
        try:
            print("Body:", resp.text)
        except Exception:
            pass
        resp.raise_for_status()

    data = resp.json()
    return data.get("response", "").strip()

def ask_hs_code(user_query: str, db: list):
    candidates = select_candidates(user_query, db, top_k=TOP_K_CANDIDATES)
    if not candidates:
        print("❌ Không có dữ liệu HS nào trong database.")
        return None

    prompt = build_prompt(user_query, candidates)
    raw_answer = call_ollama(prompt)

    try:
        parsed = json.loads(raw_answer)
    except json.JSONDecodeError:
        print("⚠ Không parse được JSON từ model, trả về raw text:")
        print(raw_answer)
        return None

    # LẤY HS CODE TỪ CANDIDATES, KHÔNG BAO GIỜ DÙNG HS CODE MODEL TỰ BỊA
    idx = parsed.get("index_in_list")
    if not isinstance(idx, int):
        print("⚠ index_in_list không phải số nguyên, bỏ qua.")
        return None

    if not (1 <= idx <= len(candidates)):
        print(f"⚠ index_in_list = {idx} nằm ngoài khoảng 1..{len(candidates)}")
        return None

    chosen = candidates[idx - 1]  # index bắt đầu từ 1

    result = {
        "hs_code": chosen.get("hs_code", ""),
        "ten_hang_trong_db": chosen.get("ten_hang", ""),
        "ghi_chu_1": chosen.get("ghi_chu_1", ""),
        "ghi_chu_2": chosen.get("ghi_chu_2", ""),
        "index_in_list": idx,
        "explanation_vi": parsed.get("explanation_vi", ""),
        "explanation_en": parsed.get("explanation_en", ""),
    }

    return result


def main():
    print("🔹 HS Code Finder dùng Ollama (KHÔNG tự tạo HS code mới)")
    print("Nhập mô tả hàng hóa bằng tiếng Việt hoặc tiếng Anh.")
    print("Gõ 'exit' để thoát.\n")

    db = load_hs_data(JSON_DB_FILE)

    while True:
        user_query = input("Mô tả hàng hóa: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "q"):
            break

        result = ask_hs_code(user_query, db)

        if result is None:
            print("❌ Không lấy được kết quả hợp lệ.\n")
            continue

        print("\n✅ Kết quả cuối cùng (sau khi đã map từ file JSON):")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
