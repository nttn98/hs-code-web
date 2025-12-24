import json
import re
from pathlib import Path
from copy import deepcopy

DATA_DIR = Path("./data")
INPUT_OUTPUT = DATA_DIR / "output.json"
INPUT_HS = DATA_DIR / "hscode_by_colletter.json"
OUTPUT_FILE = DATA_DIR / "output_merged.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clean_desc(s):
    if s is None:
        return ""
    s = str(s)
    # remove leading '- ' sequences and whitespace
    s = re.sub(r'^\s*(?:-\s*)+', '', s).strip()
    return s

def build_indexes(rows):
    """
    Build lookup maps:
      - by_clean[ma_hang_clean] -> list of rows
      - by_inferred[ma_hang_inferred] -> list of rows
    """
    by_clean = {}
    by_inferred = {}
    for r in rows:
        mc = r.get("ma_hang_clean") or r.get("ma_hang_raw") or None
        mi = r.get("ma_hang_inferred") or None
        if mc:
            by_clean.setdefault(mc, []).append(r)
        if mi:
            by_inferred.setdefault(mi, []).append(r)
    return by_clean, by_inferred

def find_row_for_code(code, by_clean, by_inferred, prefer_level=None):
    """
    Find a row by exact code in clean or inferred. Optionally prefer rows of a given v_level.
    Also fallback to prefix/suffix heuristics.
    """
    if not code:
        return None
    # exact match clean
    if code in by_clean:
        if prefer_level is None:
            return by_clean[code][0]
        for r in by_clean[code]:
            if str(r.get("v_level")) == str(prefer_level):
                return r
        return by_clean[code][0]
    # exact match inferred
    if code in by_inferred:
        if prefer_level is None:
            return by_inferred[code][0]
        for r in by_inferred[code]:
            if str(r.get("v_level")) == str(prefer_level):
                return r
        return by_inferred[code][0]
    # prefix/suffix heuristic: find any key that startswith code or code startswith key
    for k, lst in by_clean.items():
        if k.startswith(code) or code.startswith(k):
            return lst[0]
    for k, lst in by_inferred.items():
        if k.startswith(code) or code.startswith(k):
            return lst[0]
    return None

def remove_ids_recursive(obj):
    """
    Remove any key named 'id' in nested dicts/lists.
    """
    if isinstance(obj, dict):
        obj.pop("id", None)
        for k, v in list(obj.items()):
            remove_ids_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_ids_recursive(item)

def merge_chapter(ch, by_clean, by_inferred):
    # copy to avoid mutating original if needed
    ch = deepcopy(ch)

    # map description -> mo_ta_vn and try to find EN via level 0 rows
    if "description" in ch and ch.get("description"):
        ch["mo_ta_vn"] = ch.get("description")
    ch["mo_ta_en"] = ch.get("mo_ta_en", "")

    leaves = ch.get("leaf", []) or []
    new_leaves = []
    for leaf in leaves:
        # leaf_code may be in 'id' or 'ma_hang' or inside leaf children inference
        leaf_code = leaf.get("id") or leaf.get("ma_hang") or None
        leaf_name = leaf.get("name") or leaf.get("mo_ta_vn") or ""

        # if no code, try infer from children codes (take first 4)
        if not leaf_code:
            for g in leaf.get("group1", []):
                for c in g.get("children", []):
                    cc = c.get("id") or c.get("ma_hang")
                    if cc and len(str(cc)) >= 4:
                        leaf_code = str(cc)[:4]
                        break
                if leaf_code:
                    break

        # if still no code, try match by VN description among level 0 rows in by_inferred
        if not leaf_code:
            cleaned_leaf_name = clean_desc(leaf_name)
            for key, rows in by_inferred.items():
                for r in rows:
                    if str(r.get("v_level")) == "0":
                        other = r.get("other_fields") or {}
                        vn = other.get("Mô tả hàng hoá - Tiếng Việt") or r.get("mo_ta_vn")
                        if vn and clean_desc(vn) == cleaned_leaf_name:
                            leaf_code = r.get("ma_hang_inferred") or key
                            break
                if leaf_code:
                    break

        new_leaf = {
            "ma_hang": leaf_code or "",
            "mo_ta_vn": clean_desc(leaf_name),
            "mo_ta_en": ""
        }

        # try to get EN for the leaf using level 0 row
        if leaf_code:
            row0 = find_row_for_code(leaf_code, by_clean, by_inferred, prefer_level=0)
            if row0:
                other = row0.get("other_fields") or {}
                en = other.get("Mô tả hàng hoá - Tiếng Anh") or row0.get("mo_ta_en")
                if en:
                    new_leaf["mo_ta_en"] = clean_desc(en)

        # process group1
        groups_out = []
        for g in leaf.get("group1", []):
            g_name = g.get("name") or g.get("mo_ta_vn") or ""
            g_obj = {
                # no 'id' per request
                "mo_ta_vn": clean_desc(g_name),
                "mo_ta_en": "",
                "children": []
            }
            # try to find EN for group using v_level==1 rows
            candidate_group_row = None
            if new_leaf["ma_hang"]:
                for r in by_inferred.get(new_leaf["ma_hang"], []):
                    if str(r.get("v_level")) == "1":
                        other = r.get("other_fields") or {}
                        vn = other.get("Mô tả hàng hoá - Tiếng Việt") or r.get("mo_ta_vn")
                        if vn and clean_desc(vn) == clean_desc(g_name):
                            candidate_group_row = r
                            break
                if not candidate_group_row:
                    # fallback any v_level==1 under this inferred code
                    for r in by_inferred.get(new_leaf["ma_hang"], []):
                        if str(r.get("v_level")) == "1":
                            candidate_group_row = r
                            break
            if candidate_group_row:
                other = candidate_group_row.get("other_fields") or {}
                g_obj["mo_ta_en"] = clean_desc(other.get("Mô tả hàng hoá - Tiếng Anh") or candidate_group_row.get("mo_ta_en", ""))

            # children
            for child in g.get("children", []):
                child_code = child.get("id") or child.get("ma_hang") or None
                # find row
                row = find_row_for_code(child_code, by_clean, by_inferred)
                co = {}
                if row:
                    other = row.get("other_fields") or {}
                    co["ma_hang"] = row.get("ma_hang_clean") or row.get("ma_hang_inferred") or child_code or ""
                    # descriptions
                    co["mo_ta_vn"] = clean_desc(row.get("mo_ta_vn") or other.get("Mô tả hàng hoá - Tiếng Việt") or child.get("name") or "")
                    co["mo_ta_en"] = clean_desc(row.get("mo_ta_en") or other.get("Mô tả hàng hoá - Tiếng Anh") or "")
                    # copy all other_fields except Mã hàng and descriptions
                    exclude = {"Mã hàng", "Mô tả hàng hoá - Tiếng Việt", "Mô tả hàng hoá - Tiếng Anh"}
                    for k, v in other.items():
                        if k in exclude:
                            continue
                        co[k] = v
                    # also copy any top-level keys inside row that are useful and not already present
                    for topk in ("NK TT", "Văn bản", "Ngày hiệu lực"):
                        if topk in row and topk not in co:
                            co[topk] = row[topk]
                else:
                    # fallback minimal from child info
                    co["ma_hang"] = child_code or ""
                    co["mo_ta_vn"] = clean_desc(child.get("name") or child.get("mo_ta_vn") or "")
                    co["mo_ta_en"] = ""
                g_obj["children"].append(co)
            groups_out.append(g_obj)

        new_leaf["group1"] = groups_out
        new_leaves.append(new_leaf)

    ch["leaf"] = new_leaves

    # remove original description/id fields if present
    for k in ("description", "id", "name"):
        if k in ch:
            ch.pop(k, None)

    # ensure no id keys anywhere inside this chapter
    remove_ids_recursive(ch)
    return ch

def merge_all(output_data, hs_rows):
    by_clean, by_inferred = build_indexes(hs_rows)

    # decide how chapters are stored in output_data:
    # case A: output_data is list of chapters
    if isinstance(output_data, list):
        return [merge_chapter(ch, by_clean, by_inferred) for ch in output_data]

    # case B: output_data is dict representing one chapter
    if isinstance(output_data, dict):
        # if it's one chapter with leaf directly, merge it as single chapter and return as dict
        if "leaf" in output_data or "description" in output_data:
            merged = merge_chapter(output_data, by_clean, by_inferred)
            return merged
        # if it has a list of chapters under a common key (try common keys)
        for key in ("chapters", "data", "items"):
            if key in output_data and isinstance(output_data[key], list):
                output_data[key] = [merge_chapter(ch, by_clean, by_inferred) for ch in output_data[key]]
                return output_data
        # fallback: try to detect any value that looks like a chapter and merge
        changed = False
        for k, v in list(output_data.items()):
            if isinstance(v, dict) and ("leaf" in v or "description" in v):
                output_data[k] = merge_chapter(v, by_clean, by_inferred)
                changed = True
            elif isinstance(v, list) and v and isinstance(v[0], dict) and ("leaf" in v[0] or "description" in v[0]):
                output_data[k] = [merge_chapter(ch, by_clean, by_inferred) for ch in v]
                changed = True
        if changed:
            return output_data

    raise ValueError("Unexpected structure of output.json — cannot find chapters/leaves")

def main():
    if not INPUT_OUTPUT.exists():
        print(f"Input file not found: {INPUT_OUTPUT}")
        return
    if not INPUT_HS.exists():
        print(f"HS input file not found: {INPUT_HS}")
        return

    out_json = load_json(INPUT_OUTPUT)
    hs_json = load_json(INPUT_HS)

    # ensure hs rows is a list (if top-level is dict, try to locate list)
    rows = hs_json
    if isinstance(hs_json, dict):
        # try find a list value
        found_list = None
        for v in hs_json.values():
            if isinstance(v, list):
                found_list = v
                break
        if found_list is None:
            raise ValueError("hs_code_colletter.json does not contain list of rows at top-level")
        rows = found_list

    merged = merge_all(out_json, rows)

    # remove any 'id' keys globally to be safe
    remove_ids_recursive(merged)

    save_json(merged, OUTPUT_FILE)
    print(f"Merge completed -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
