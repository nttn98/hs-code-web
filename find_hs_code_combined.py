import json
import os
import unicodedata
import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load GROQ_API_KEY from .env

_RE_COMBINING = re.compile(r"[\u0300-\u036f]")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]", flags=re.I)

VIETNAMESE_STOPWORDS = {
    'cái', 'con', 'chiếc', 'cục', 'hộp', 'bộ', 'món', 'thứ',
    'những', 'nhung', 'các', 'cac', 'một', 'mot', 'hai', 'ba',
    'là', 'la', 'của', 'cua', 'cho', 'với', 'voi', 'và', 'va',
    'được', 'duoc', 'có', 'co', 'bằng', 'bang', 'từ', 'tu',
    'để', 'de', 'này', 'nay', 'đó', 'do', 'kia'
}

def normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFD", s)
    s = _RE_COMBINING.sub("", s)
    s = s.lower()
    s = _RE_NON_ALNUM.sub(" ", s)
    return " ".join([w for w in s.split() if w not in VIETNAMESE_STOPWORDS])

def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

class HSFinder:
    def __init__(self, data_dir: str = "data"):
        self.data_json_path = os.path.join(data_dir, "output.json")
        self.roots = []
        self.tree_map = {}
        self._load_data()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _load_data(self):
        if not os.path.exists(self.data_json_path):
            raise FileNotFoundError(f"{self.data_json_path} not found")
        with open(self.data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.roots = data.get("roots", [])
        self.tree_map = {}
        for root in self.roots:
            root_name = root.get("name", "")
            root_desc = root.get("description", "")
            for leaf in root.get("leaf", []):
                for g1 in leaf.get("group1", []):
                    children = g1.get("children", [g1])
                    for child in children:
                        code = child.get("id") or child.get("ma_hang")
                        name = child.get("name") or ""
                        if code:
                            self.tree_map[code] = {
                                "name": name,
                                "root_name": root_name,
                                "root_desc": root_desc
                            }

    def find(self, query: str, top_n: int = 3) -> List[Dict[str, str]]:
        q_norm = normalize(query)
        results = []

        for code, info in self.tree_map.items():
            leaf_name = normalize(info.get("name", ""))
            root_desc = normalize(info.get("root_desc", ""))
            score = 0.0
            if q_norm in leaf_name:
                score += 1.0
            if q_norm in root_desc:
                score += 0.5
            score += fuzzy_score(q_norm, leaf_name) * 0.5
            if score > 0:
                results.append({"code": code, "name": info.get("name", ""), "score": score})

        results.sort(key=lambda x: -x["score"])

        filtered = []
        for r in results:
            name_l = r["name"].lower()
            if any(bad in name_l for bad in ["dạng", "chế phẩm"]) and q_norm not in name_l:
                continue
            filtered.append({"code": r["code"], "name": r["name"]})
            if len(filtered) >= top_n:
                break

        return filtered

    def query_groq(self, query_text: str, top_n: int = 3) -> List[Dict[str, str]]:
        """
        Use Groq to understand query and return HS codes from output.json
        Limit message length to avoid BadRequestError
        """
        messages = [
            {
                "role": "user",
                "content": query_text[:500]  # truncate to 500 chars to reduce token usage
            }
        ]
        results = []
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=1,
                max_completion_tokens=256,
                top_p=1
            )
            content = completion.choices[0].message.content
            # Expect Groq to return HS code list as JSON string
            try:
                parsed = json.loads(content)
                for item in parsed:
                    code = item.get("code")
                    name = item.get("name")
                    if code and name:
                        results.append({"code": code, "name": name})
                        if len(results) >= top_n:
                            break
            except Exception:
                # fallback: use local find
                results = self.find(query_text, top_n=top_n)
        except Exception:
            results = self.find(query_text, top_n=top_n)

        return results
