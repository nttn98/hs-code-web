import json
import pandas as pd
import hashlib
import os

INPUT_EXCEL = "./data/data.xls"   # hoặc .xlsx
OUTPUT_JSON = "./data/output.json"
HASH_FILE = "./data/data.xls.hash"


def file_hash(path):
    """Tính SHA256 hash của file"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_file_changed(path, hash_file):
    """Kiểm tra file có thay đổi không"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")

    new_hash = file_hash(path)

    if not os.path.exists(hash_file):
        # lần đầu chạy
        with open(hash_file, "w") as f:
            f.write(new_hash)
        return True

    with open(hash_file, "r") as f:
        old_hash = f.read().strip()

    if new_hash != old_hash:
        with open(hash_file, "w") as f:
            f.write(new_hash)
        return True

    return False


def excel_to_flat_json(path):
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    result = []

    for _, row in df.iterrows():
        ten_hang = "" if pd.isna(row["tên hàng"]) else str(row["tên hàng"]).strip()
        hs_code = (
            str(row["hs code"]).strip()
            if not pd.isna(row["hs code"]) and str(row["hs code"]).strip()
            else None
        )
        chinh_sach = (
            "" if pd.isna(row["chính sách mặt hàng"])
            else str(row["chính sách mặt hàng"]).strip()
        )

        result.append({
            "hs_code": hs_code,
            "ten_hang": ten_hang.rstrip(":"),
            "chinh_sach": chinh_sach
        })

    return result


if __name__ == "__main__":
    if not is_file_changed(INPUT_EXCEL, HASH_FILE):
        print("⏩ data.xls không thay đổi → bỏ qua export")
        exit(0)

    data = excel_to_flat_json(INPUT_EXCEL)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ data.xls đã thay đổi → Export JSON thành công")
