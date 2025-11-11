# hs_loader.py
import os
import pandas as pd

DATA_DIR = "data"


def _load_json_goods(path: str) -> pd.DataFrame:
    """
    Đọc file JSON kiểu:
      [
        {
          "ten_hang": "...",
          "hs_code": "32082090",
          "ghi_chu_1": "...",
          "ghi_chu_2": "...",
          "item_code": "...",
          "text": "Tên hàng: ... | Ghi chú 1: ... | Ghi chú 2: ... | item_code: ..."
        }
      ]

    Kết quả:
      - hs_code
      - description : chuỗi đầy đủ (tên hàng + ghi chú + item...)
      - ten_hang    : chỉ tên hàng (nếu có)
      - source_file
    """
    try:
        df = pd.read_json(path, orient="records")
    except ValueError:
        df = pd.read_json(path, lines=True)

    # Map lowercase -> tên cột gốc
    cols_lower = {c.lower(): c for c in df.columns}

    # Cột hs_code (bắt buộc)
    hs_col = next(
        (cols_lower[k] for k in cols_lower if "hs_code" in k or "hs code" in k),
        None,
    )
    if not hs_col:
        raise ValueError(f"File {os.path.basename(path)} không có cột hs_code.")

    # Cột tên hàng (tùy chọn)
    name_col = next(
        (cols_lower[k] for k in cols_lower if "ten_hang" in k or "tên hàng" in k or "ten hang" in k),
        None,
    )

    # Nếu có field 'text' đã gộp sẵn thì ưu tiên
    text_col = cols_lower.get("text")

    # 1) hs_code
    hs_code = df[hs_col].astype(str).fillna("").str.strip()

    # 2) ten_hang (nếu có cột)
    if name_col:
        ten_hang = df[name_col].astype(str).fillna("").str.strip()
    else:
        ten_hang = pd.Series([""] * len(df))

    # 3) description: full thông tin để LLM dùng
    if text_col:
        desc_series = df[text_col].astype(str).fillna("").str.strip()
    else:
        # bắt đầu từ tên hàng (nếu có)
        desc_series = ten_hang.copy()

        # gom các cột có chứa “ghi”, “note”, “item”, “remark” (trừ hs)
        for c in df.columns:
            c_low = str(c).lower()
            if ("hs" in c_low):
                continue
            if any(k in c_low for k in ["ghi", "note", "item", "remark"]):
                desc_series = desc_series + f" | {c}: " + df[c].astype(str).fillna("").str.strip()

    desc_series = desc_series.astype(str).fillna("").str.strip()

    # 4) DataFrame kết quả
    result = pd.DataFrame(
        {
            "hs_code": hs_code,
            "description": desc_series,      # cho LLM đọc
            "ten_hang": ten_hang,            # cho UI hiển thị
            "source_file": os.path.basename(path),
        }
    )

    # Bỏ dòng trống
    result = result[
        (result["hs_code"].str.strip() != "")
        & (result["description"].str.strip() != "")
    ]

    return result


def load_hs_excels(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc tất cả file JSON trong thư mục data/
    và gom thành 1 DataFrame duy nhất.
    """
    all_rows = []

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".json"):
            continue

        path = os.path.join(data_dir, fname)
        print(f"Loading HS JSON file: {path}")
        try:
            df_part = _load_json_goods(path)
        except Exception as e:
            print(f"⚠️ Bỏ qua file {fname}: {e}")
            continue

        all_rows.append(df_part)

    if not all_rows:
        raise ValueError("Không tìm thấy file JSON HS code nào trong folder data/")

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df["hs_code"] = full_df["hs_code"].astype(str).str.strip()
    full_df["description"] = full_df["description"].astype(str).str.strip()
    full_df["ten_hang"] = full_df["ten_hang"].astype(str).fillna("").str.strip()

    return full_df
