import json
from pathlib import Path
import pandas as pd
import math

# ================== CONFIG ==================
EXCEL_FILE = "data/grdData.xlsx"       # file excel của bạn
JSON_OUTPUT = "hs_codes.json"          # file json xuất ra


def safe_str(v):
    """Đổi giá trị sang string, nếu NaN/None thì trả về ''. """
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def build_text(row):
    """
    Tạo field 'text' cho LLM/Ollama đọc:
    - Ưu tiên TÊN HÀNG + GHI CHÚ 1 + GHI CHÚ 2
    - Không cần HS CODE ở đây (vì đó là đáp án)
    """
    ten_hang = safe_str(row.get("TÊN HÀNG", ""))
    ghi_chu_1 = safe_str(row.get("GHI CHÚ 1", ""))
    ghi_chu_2 = safe_str(row.get("GHI CHÚ 2", ""))

    parts = []
    if ten_hang:
        parts.append(f"Tên hàng: {ten_hang}")
    if ghi_chu_1:
        parts.append(f"Ghi chú 1: {ghi_chu_1}")
    if ghi_chu_2:
        parts.append(f"Ghi chú 2: {ghi_chu_2}")

    return " | ".join(parts)


def main():
    excel_path = Path(EXCEL_FILE)
    if not excel_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {excel_path.resolve()}")

    # Đọc file Excel (.xlsx)
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Xoá dòng trống
    df = df.dropna(how="all")

    records = []
    for _, row in df.iterrows():
        item = {
            "ten_hang": safe_str(row.get("TÊN HÀNG", "")),
            "hs_code": safe_str(row.get("HS CODE", "")),
            "ghi_chu_1": safe_str(row.get("GHI CHÚ 1", "")),
            "ghi_chu_2": safe_str(row.get("GHI CHÚ 2", "")),
            "nguoi_tao": safe_str(row.get("Người Tạo", "")),
            "ngay_tao": safe_str(row.get("Ngày Tạo", "")),
        }

        # Bỏ dòng rỗng
        if not item["ten_hang"] and not item["hs_code"]:
            continue

        item["text"] = build_text(row)
        records.append(item)

    # Ghi ra file JSON
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã ghi {len(records)} dòng vào {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
