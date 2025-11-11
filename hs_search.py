# hs_search.py
from typing import List
import pandas as pd
import re
import difflib


def _tokenize(text: str) -> List[str]:
    """
    Tách text thành token chữ/số, viết thường.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return re.findall(r"\w+", text)


def find_candidate_rows(df: pd.DataFrame, query: str, top_k: int = 50) -> pd.DataFrame:
    """
    Tìm các dòng HS code ứng viên cho mô tả hàng hóa (query).

    Nguyên tắc:
    - Điểm = kết hợp:
        + tỉ lệ token query xuất hiện trong description
        + độ giống chuỗi (SequenceMatcher)
    - Ưu tiên mã HS chi tiết hơn: code_len (số chữ số sau khi bỏ dấu chấm) càng dài càng tốt.
    - Không hard-code ngành.
    """

    if df is None or df.empty:
        return df

    tmp = df.copy()

    # Chuẩn hóa mã HS thành chỉ còn số + độ dài
    tmp["digits"] = tmp["hs_code"].astype(str).str.replace(r"\D", "", regex=True)
    tmp["code_len"] = tmp["digits"].str.len()

    query_tokens = _tokenize(query)
    query_token_set = set(query_tokens)
    query_lower = str(query).lower()

    # Nếu user không nhập nội dung, fallback: trả những mã chi tiết nhất
    if not query_token_set:
        tmp_sorted = tmp.sort_values(by=["code_len"], ascending=False)
        max_len = tmp_sorted["code_len"].max()
        detailed = tmp_sorted[tmp_sorted["code_len"] == max_len]
        return detailed.head(top_k)

    def score_row(desc: str) -> float:
        """
        Điểm = 2 * tỉ lệ token khớp + độ giống chuỗi (0..1).
        """
        desc_str = str(desc)
        tokens = _tokenize(desc_str)
        token_set = set(tokens)

        # tỉ lệ token query xuất hiện trong mô tả
        common = query_token_set.intersection(token_set)
        token_ratio = len(common) / max(len(query_token_set), 1)

        # độ giống chuỗi tổng thể
        char_ratio = difflib.SequenceMatcher(None, query_lower, desc_str.lower()).ratio()

        return 2.0 * token_ratio + char_ratio

    tmp["score"] = tmp["description"].apply(score_row)

    # Nếu tất cả score = 0, vẫn chọn theo code_len
    if tmp["score"].max() <= 0:
        tmp_sorted = tmp.sort_values(by=["code_len"], ascending=False)
        max_len = tmp_sorted["code_len"].max()
        detailed = tmp_sorted[tmp_sorted["code_len"] == max_len]
        return detailed.head(top_k)

    candidates = tmp[tmp["score"] > 0]

    # Nếu vì lý do gì đó rỗng, fallback dùng toàn bộ tmp
    if candidates.empty:
        candidates = tmp

    # Lấy độ dài lớn nhất trong nhóm candidate (ưu tiên 8 số nếu có)
    max_len = candidates["code_len"].max()
    detailed = candidates[candidates["code_len"] == max_len]

    # Sắp xếp:
    #   1) score giảm dần (match tốt hơn)
    #   2) code_len giảm dần (chi tiết hơn)
    #   3) digits tăng dần (ổn định)
    detailed_sorted = detailed.sort_values(
        by=["score", "code_len", "digits"],
        ascending=[False, False, True],
    )

    return detailed_sorted.head(top_k)
