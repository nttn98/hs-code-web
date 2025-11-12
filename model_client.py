# model_client.py
import os
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Chưa cấu hình GROQ_API_KEY trong file .env")

# Mặc định dùng model nhẹ hơn để tiết kiệm token
MODEL_NAME = os.environ.get("MODEL_NAME")
client = Groq(api_key=GROQ_API_KEY)


# -------------------------
# Helpers: parsing / rules
# -------------------------
def format_hs_code(hs: str) -> str:
    digits = re.sub(r"\D", "", str(hs))
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    if len(digits) == 6:
        return f"{digits[:4]}.{digits[4:]}"
    return str(hs)


def parse_thickness_from_text(text: str) -> List[Tuple[Optional[str], float]]:
    """
    Tìm các biểu diễn độ dày trong text, trả về list các tuple (comparator, value_mm)
    comparator có thể là '<', '<=', '>', '>=', '==' hoặc None (nếu chỉ số đơn).
    Ví dụ nhận được:
      "dưới 4mm" -> [('<', 4.0)]
      "4 mm" -> [(None, 4.0)]
    """
    text = str(text).lower()
    results = []

    # Các mẫu như "dưới 4mm", "ít hơn 4 mm", "<4mm", "≤5 mm", ">= 10mm"
    patterns = [
        (r"(dưới|<|ít hơn|không quá|<=|≤)\s*([0-9]+(?:[\.,][0-9]+)?)\s*mm", "<="),
        (r"(trên|>|lớn hơn|>=|≥)\s*([0-9]+(?:[\.,][0-9]+)?)\s*mm", ">="),
        (r"([0-9]+(?:[\.,][0-9]+)?)\s*-\s*([0-9]+(?:[\.,][0-9]+)?)\s*mm", "range"),
        (r"([0-9]+(?:[\.,][0-9]+)?)\s*mm", "=="),
    ]

    for pat, op in patterns:
        for m in re.finditer(pat, text):
            if op == "range":
                a = float(m.group(1).replace(",", "."))
                b = float(m.group(2).replace(",", "."))
                # represent range as two tuples: >=a and <=b
                results.append((">=", a))
                results.append(("<= ", b))
            else:
                try:
                    val = float(m.group(2).replace(",", "."))
                except Exception:
                    val = float(m.group(1).replace(",", "."))
                # map operator labels to symbolic comparators
                if op == "<=":
                    comp = "<="
                elif op == ">=":
                    comp = ">="
                elif op == "==":
                    comp = "=="
                else:
                    comp = None
                results.append((comp, val))

    # also catch patterns like "dưới 4" without mm but context contains 'dày' nearby
    for m in re.finditer(r"(dày|độ dày).{0,20}?([0-9]+(?:[\.,][0-9]+)?)\s*(mm)?", text):
        try:
            val = float(m.group(2).replace(",", "."))
            results.append((None, val))
        except:
            pass

    return results


def parse_specs_from_query(query: str) -> Dict:
    """
    Trích thông số quan trọng từ câu mô tả: material, thickness comparator/value, keywords.
    Trả về dict như:
      { "material": ["mdf", "gỗ"], "thickness": [('<', 4.0), ...], "keywords": ["trẻ em"] }
    """
    q = str(query).lower()
    specs = {"material": [], "thickness": [], "keywords": []}

    # materials (expandable)
    mats = ["mdf", "ván mdf", "ván", "gỗ", "fiberboard", "hdf", "plywood", "veneer"]
    for m in mats:
        if m in q:
            specs["material"].append(m)

    # keywords
    kws = ["trẻ em", "trẻ-em", "child", "baby", "giày", "trẻ em"]  # add as needed
    for k in kws:
        if k in q:
            specs["keywords"].append(k)

    # thickness
    specs["thickness"] = parse_thickness_from_text(q)

    return specs


def extract_numbers_from_text(text: str) -> List[float]:
    """Trích tất cả số (mm) có khả năng là độ dày/kích thước."""
    nums = []
    for m in re.finditer(r"([0-9]+(?:[\.,][0-9]+)?)\s*mm", str(text).lower()):
        try:
            nums.append(float(m.group(1).replace(",", ".")))
        except:
            pass
    # fallback: bare numbers (may be risky)
    return nums


def candidate_matches_specs(row: Dict, specs: Dict) -> Tuple[bool, int]:
    """
    Kiểm tra candidate có match với specs.
    Trả về (match_bool, score) — score càng cao càng match tốt.
    Logic:
      - +10 nếu material xuất hiện trong description/ten_hang
      - +20 nếu có độ dày match (so sánh theo comparator)
      - +5 nếu keyword xuất hiện
      - +1 nếu candidate chứa bất kỳ số nào (gợi ý có thông tin kích thước)
    """
    desc_fields = []
    if row.get("ten_hang"):
        desc_fields.append(str(row.get("ten_hang", "")).lower())
    if row.get("description"):
        desc_fields.append(str(row.get("description", "")).lower())
    desc = " | ".join(desc_fields)

    score = 0
    matched = False

    # material
    for m in specs.get("material", []):
        if m and m in desc:
            score += 10
            matched = True

    # keywords
    for k in specs.get("keywords", []):
        if k and k in desc:
            score += 5
            matched = True

    # numbers in candidate
    cand_nums = extract_numbers_from_text(desc)
    if cand_nums:
        score += 1

    # thickness matching: if specs has comparators, try to evaluate
    th_specs = specs.get("thickness", [])
    if th_specs:
        for comp, val in th_specs:
            for cand_val in cand_nums:
                try:
                    if comp in ("<", "<=") or comp is None and comp != ">=":
                        # treat None/== as equality-ish; but if user said "dưới 4mm" comp is '<='
                        if cand_val <= val:
                            score += 20
                            matched = True
                            break
                    if comp in (">", ">="):
                        if cand_val >= val:
                            score += 20
                            matched = True
                            break
                    if comp == "==":
                        if abs(cand_val - val) < 1e-6:
                            score += 20
                            matched = True
                            break
                except Exception:
                    pass

            # if candidate has no explicit numbers but description contains textual ranges, try substring match
            if not cand_nums:
                # look for phrases like 'không quá 5 mm' or 'trên 9 mm' in desc
                if comp == "<=" and re.search(r"(không quá|dưới|ít hơn|<=|≤)\s*%s\s*mm" % int(val), desc):
                    score += 20
                    matched = True
                if comp == ">=" and re.search(r"(trên|lớn hơn|>=|≥)\s*%s\s*mm" % int(val), desc):
                    score += 20
                    matched = True

    return matched, score


def filter_candidates_by_specs(candidates: List[Dict], specs: Dict) -> List[Dict]:
    """
    Lọc và sắp xếp candidate dựa trên specs.
    Trả về danh sách đã sort theo score giảm dần.
    Nếu tất cả score==0 thì trả về nguyên list (không lọc).
    """
    scored = []
    for row in candidates:
        matched, score = candidate_matches_specs(row, specs)
        scored.append((score, row))

    # sort desc
    scored.sort(key=lambda x: x[0], reverse=True)

    # if top score is 0 -> no useful info, return original
    if scored and scored[0][0] == 0:
        return candidates

    # else return rows with score > 0 (but keep a few top ones)
    filtered = [r for s, r in scored if s > 0]
    # if filtered is empty (shouldn't), fallback
    if not filtered:
        return candidates
    # limit top N to reduce prompt size
    return filtered[:12]


# -------------------------
# LLM call + main entry
# -------------------------
def _extract_hs_from_output(text: str) -> Optional[str]:
    m = re.search(r"\b\d{4}(?:\.\d{2}(?:\.\d{2})?)?\b", text)
    return m.group(0) if m else None


def build_context_for_llm(candidates: List[Dict]) -> str:
    """
    Format ngắn gọn hơn cho LLM: mỗi dòng 'idx. HS: xxxx | short desc'
    """
    lines = []
    for i, row in enumerate(candidates, start=1):
        raw = row.get("hs_code", "")
        desc = (row.get("ten_hang") or row.get("description") or "")[:140]
        lines.append(f"{i}. HS: {raw} | {desc}")
    return "\n".join(lines)


def ask_model_for_hs(query: str, candidate_rows) -> str:
    """
    Optimized version: compact context + top_k candidates + deterministic shortcut by score margin.
    Keeps original behavior & fallbacks.
    """
    # convert dataframe->records
    candidates = candidate_rows.to_dict(orient="records")
    if not candidates:
        return "Không tìm được mã HS ứng viên nào."

    specs = parse_specs_from_query(query)
    filtered = filter_candidates_by_specs(candidates, specs)

    # If filter yields single candidate, return it deterministically
    if len(filtered) == 1:
        chosen = filtered[0]
        formatted = format_hs_code(chosen.get("hs_code", ""))
        names = []
        for r in [chosen]:
            n = r.get("ten_hang") or r.get("description") or ""
            n = str(n).strip()
            if n and n not in names:
                names.append(n)
        lines = [f"HS code: {formatted}"]
        if names:
            lines.append("Các tên hàng khớp trong dữ liệu:")
            for name in names[:8]:
                lines.append(f"- {name}")
        return "\n".join(lines)

    # --- Scoring top candidates to possibly avoid call ---
    # compute scores for filtered candidates (use candidate_matches_specs)
    scored_list = []
    for row in filtered:
        matched, score = candidate_matches_specs(row, specs)
        scored_list.append((score, row))
    # sort desc
    scored_list.sort(key=lambda x: x[0], reverse=True)
    if not scored_list:
        return "Không tìm được mã HS ứng viên phù hợp."

    # if top candidate is clearly better than second by margin -> pick deterministically
    score_margin = 15  # threshold: điều chỉnh nếu muốn
    top_score = scored_list[0][0]
    second_score = scored_list[1][0] if len(scored_list) > 1 else -9999
    if top_score >= second_score + score_margin:
        chosen = scored_list[0][1]
        formatted = format_hs_code(chosen.get("hs_code", ""))
        matched_rows = [r for r in candidates if re.sub(r"\D", "", str(r.get("hs_code", ""))) == re.sub(r"\D", "", str(chosen.get("hs_code", "")))]
        names = []
        for r in matched_rows:
            n = r.get("ten_hang") or r.get("description") or ""
            n = str(n).strip()
            if n and n not in names:
                names.append(n)
        lines = [f"HS code: {formatted}"]
        if names:
            lines.append("Các tên hàng khớp trong dữ liệu:")
            for name in names[:12]:
                lines.append(f"- {name}")
        return "\n".join(lines)

    # --- Build compact context for LLM ---
    def build_compact_context_for_llm(candidates_list: List[Dict], max_chars_desc: int = 40, top_k: int = 6):
        top = candidates_list[:top_k]
        lines = []
        for i, row in enumerate(top, start=1):
            hs_raw = re.sub(r"\D", "", str(row.get("hs_code", "")))
            desc = (row.get("ten_hang") or row.get("description") or "")[:max_chars_desc].replace("\n", " ").strip()
            text = (row.get("ten_hang") or row.get("description") or "").lower()
            tags = []
            for m in ["mdf", "hdf", "plywood", "veneer", "gỗ", "ván"]:
                if m in text:
                    tags.append(m)
            nums = []
            for m in re.finditer(r"([0-9]+(?:[\.,][0-9]+)?)\s*mm", text):
                nums.append(m.group(1).replace(",", "."))
            nums_str = nums[0] if nums else ""
            tag_str = ",".join(tags) if tags else ""
            lines.append(f"{i}|HS={hs_raw}|d={desc}|n={nums_str}|t={tag_str}")
        return "\n".join(lines), top

    def build_specs_summary_local(specs: Dict) -> str:
        parts = []
        mats = specs.get("material", [])
        if mats:
            parts.append("material:" + ",".join(mats[:3]))
        th = specs.get("thickness", [])
        if th:
            comp, val = th[0]
            if comp:
                parts.append(f"thickness:{comp}{val}")
            else:
                parts.append(f"thickness:=={val}")
        kws = specs.get("keywords", [])
        if kws:
            parts.append("kw:" + ",".join(kws[:3]))
        return ";".join(parts)

    context_str, top_candidates = build_compact_context_for_llm([r for s, r in scored_list], max_chars_desc=40, top_k=6)
    specs_summary = build_specs_summary_local(specs)

    system_prompt = "Bạn là chuyên gia phân loại mã HS Việt Nam. Chỉ chọn 1 mã HS từ danh sách đã cung cấp. Trả 1 dòng: HS code: <mã>"

    user_prompt = (
        f"Mô tả hàng: \"{query}\"\n"
        f"Specs: {specs_summary}\n\n"
        "Danh sách ứng viên (compact):\n"
        f"{context_str}\n\n"
        "LƯU Ý: CHỈ CHỌN 1 mã HS từ danh sách trên. Trả DUY NHẤT 1 dòng: HS code: <mã>"
    )

    # call LLM (compact prompt -> fewer tokens). keep temp=0 for determinism
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=40,
        )
        model_output = completion.choices[0].message.content or ""
    except RateLimitError:
        return "🚫 Hết giới hạn Groq trong ngày, vui lòng thử lại sau."
    except Exception as e:
        # fallback: choose top filtered candidate if exists
        if filtered:
            chosen = filtered[0]
            formatted = format_hs_code(chosen.get("hs_code", ""))
            return f"HS code: {formatted}"
        return f"⚠️ Lỗi khi gọi model: {e}"

    # parse model output and map back to candidate
    chosen_text = _extract_hs_from_output(model_output)
    if not chosen_text:
        # fallback to top filtered
        if filtered:
            chosen = filtered[0]
        else:
            chosen = candidates[0]
    else:
        chosen_digits = re.sub(r"\D", "", chosen_text)
        chosen = None
        # search in top candidates first
        for r in top_candidates:
            if re.sub(r"\D", "", str(r.get("hs_code", ""))) == chosen_digits:
                chosen = r
                break
        # then search in filtered
        if not chosen:
            for r in filtered:
                if re.sub(r"\D", "", str(r.get("hs_code", ""))) == chosen_digits:
                    chosen = r
                    break
        # finally search in all candidates
        if not chosen:
            for r in candidates:
                if re.sub(r"\D", "", str(r.get("hs_code", ""))) == chosen_digits:
                    chosen = r
                    break
        if not chosen:
            chosen = filtered[0] if filtered else candidates[0]

    formatted = format_hs_code(chosen.get("hs_code", ""))
    matched_rows = [r for r in candidates if re.sub(r"\D", "", str(r.get("hs_code", ""))) == re.sub(r"\D", "", str(chosen.get("hs_code", "")))]
    names = []
    for r in matched_rows:
        n = r.get("ten_hang") or r.get("description") or ""
        n = str(n).strip()
        if n and n not in names:
            names.append(n)

    lines = [f"HS code: {formatted}"]
    if names:
        lines.append("Các tên hàng khớp trong dữ liệu:")
        for name in names[:12]:
            lines.append(f"- {name}")

    return "\n".join(lines)
