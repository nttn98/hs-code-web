# services/hs_external.py
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def fetch_caselaw_hierarchy(hs_code):
    if not hs_code:
        return {"chapter": "", "chapter_groups": {}}

    url = f"https://caselaw.vn/ket-qua-tra-cuu-ma-hs?query={quote_plus(str(hs_code))}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except:
        return {"chapter": "", "chapter_groups": {}}

    soup = BeautifulSoup(resp.text, "html.parser")
    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]

    chapter_re = re.compile(r'^(Chương\s+\d+)\s*-?\s*(.*)$', re.I)
    code_re = re.compile(r'^(\d{4,10})\s*-?\s*(.*)$')

    chapters = []
    groups = {}
    current = ""

    for i, ln in enumerate(lines):
        ch = chapter_re.match(ln)
        if ch:
            title = f"{ch.group(1)} – {ch.group(2)}".strip(" –")
            chapters.append(title)
            current = title
            groups.setdefault(current, [])
            continue

        cd = code_re.match(ln)
        if cd and current:
            label = f"{cd.group(1)} – {cd.group(2)}".strip(" –")
            groups[current].append(label)

    return {
        "chapter": chapters[0] if chapters else "",
        "chapter_groups": groups
    }
