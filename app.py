import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

from hs_loader import load_hs_excels
from hs_search import find_candidate_rows
from model_client import ask_model_for_hs

app = Flask(__name__)

HS_DF = load_hs_excels()


def extract_hs_from_result(result: str) -> str | None:
    """Lấy mã HS từ 'HS code: 4411.13.00' -> '44111300'."""
    if not result:
        return None
    m = re.search(r"\b\d{4}(?:\.\d{2}(?:\.\d{2})?)?\b", result)
    if not m:
        return None
    return re.sub(r"\D", "", m.group(0))

def fetch_caselaw_hierarchy(hs_digits: str) -> dict:
    """
    Gọi Caselaw.vn để lấy thông tin phân cấp có kèm mô tả, ví dụ:

    {
        "chapter": "Chương 32 – Các chất nhuộm hữu cơ; ...",
        "groups": [
            "3208 – Sơn và véc-ni dựa trên polyme ...",
            "32082090 – Loại khác"
        ]
    }
    """
    url = f"https://caselaw.vn/ket-qua-tra-cuu-ma-hs?query={hs_digits}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Lỗi gọi Caselaw:", e)
        return {"chapter": "", "groups": []}

    soup = BeautifulSoup(resp.text, "html.parser")
    raw_text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    chapter = ""
    groups_map: dict[str, str] = {}

    # HS dạng 4 / 6 / 8 số, dùng để lọc những line liên quan
    hs4 = hs_digits[:4]
    hs6 = hs_digits[:6] if len(hs_digits) >= 6 else None
    hs8 = hs_digits[:8] if len(hs_digits) >= 8 else None

    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i]

        # 1) Chương
        if ln.startswith("Chương") and not chapter:
            desc = ""
            # Thử ghép thêm dòng dưới nếu là text mô tả
            if i + 1 < n:
                nxt = lines[i + 1]
                if not nxt.startswith("Chương") and not re.match(r"^\d{4,8}$", nxt):
                    desc = nxt
            chapter = ln if not desc else f"{ln} – {desc}"
            i += 1
            continue

        # 2) Nhóm 4/6/8 số
        m = re.match(r"^(\d{4,8})\b(.*)$", ln)
        if m:
            code = m.group(1)
            tail = m.group(2).strip(" -–:\u2013")  # bỏ dấu nối, cách, etc.

            # chỉ giữ code liên quan đến HS hiện tại
            if code not in {hs4, hs6, hs8}:
                i += 1
                continue

            desc = tail

            # nếu cùng dòng không có mô tả -> kiểm tra dòng dưới
            if not desc and i + 1 < n:
                nxt = lines[i + 1]
                if (
                    not nxt.startswith("Chương")
                    and not re.match(r"^\d{4,8}$", nxt)
                ):
                    desc = nxt

            if desc:
                groups_map[code] = f"{code} – {desc}"
            else:
                groups_map[code] = code

        i += 1

    # Sắp theo thứ tự 4 -> 6 -> 8
    groups = []
    for code in [hs4, hs6, hs8]:
        if code and code in groups_map:
            groups.append(groups_map[code])

    return {"chapter": chapter, "groups": groups}

@app.route("/", methods=["GET", "POST"])
def index():
    user_query = ""
    result = ""
    caselaw_data = {}

    if request.method == "POST":
        user_query = request.form.get("description", "").strip()
        if user_query:
            # 1. Tìm ứng viên
            candidates = find_candidate_rows(HS_DF, user_query)

            # 2. Hỏi model
            result = ask_model_for_hs(user_query, candidates)

            # 3. Gọi Caselaw để lấy phân cấp
            hs_digits = extract_hs_from_result(result)
            if hs_digits:
                caselaw_data = fetch_caselaw_hierarchy(hs_digits)

    return render_template(
        "index.html",
        user_query=user_query,
        result=result,
        caselaw_data=caselaw_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
