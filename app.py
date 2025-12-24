# app.py
import os
import re
import json
from urllib.parse import quote_plus
import requests
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
client = Groq(api_key=api_key)

app = Flask(__name__)

DATA_DIR = os.path.join("data")
HSCODE_JSON = os.path.join(DATA_DIR, "output.json")

# Load và parse output.json
def load_hs_database():
    """Parse output.json thành list flat của HS codes"""
    try:
        with open(HSCODE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return parse_hierarchy(data)
    except FileNotFoundError:
        print("❌ Error: output.json không tìm thấy")
        return []
    except json.JSONDecodeError:
        print("❌ Error: output.json có lỗi format")
        return []

def parse_hierarchy(data):
    """Chuyển cấu trúc phân cấp thành list flat"""
    results = []
    
    def traverse(node, chapter_info=""):
        # Xử lý node có code
        if 'code' in node and node['code']:
            code = str(node['code']).strip()
            # Chỉ lấy code là số thuần
            if code.isdigit() and len(code) >= 4:
                name = node.get('name', '').strip()
                if name:  # Chỉ lấy code có description
                    results.append({
                        'code': code,
                        'name': name,
                        'chapter': chapter_info
                    })
        
        # Traverse children
        if 'children' in node:
            for child in node['children']:
                traverse(child, chapter_info)
        
        # Traverse group1
        if 'group1' in node:
            for group in node['group1']:
                traverse(group, chapter_info)
        
        # Traverse leaf
        if 'leaf' in node:
            for leaf in node['leaf']:
                traverse(leaf, chapter_info)
    
    # Bắt đầu từ roots
    if 'roots' in data:
        for root in data['roots']:
            chapter = f"Chương {root.get('code', '')} - {root.get('description', '')}"
            traverse(root, chapter)
    
    return results

HS_DATABASE = load_hs_database()

def smart_filter_candidates(user_query, max_results=50):
    """
    Tìm kiếm thông minh các HS code ứng viên dựa trên từ khóa
    """
    if not HS_DATABASE:
        return []
    
    query_lower = user_query.lower().strip()
    query_words = set(re.findall(r'\w+', query_lower))
    
    candidates = []
    
    for item in HS_DATABASE:
        name_lower = item['name'].lower()
        name_words = set(re.findall(r'\w+', name_lower))
        
        # Tính điểm match
        score = 0
        
        # 1. Exact phrase match (điểm cao nhất)
        if query_lower in name_lower:
            score += 100
        
        # 2. Đếm số từ khớp
        matching_words = query_words.intersection(name_words)
        score += len(matching_words) * 10
        
        # 3. Thứ tự từ khớp (bonus nếu từ đầu tiên match)
        if query_words and name_words:
            first_query_word = list(query_words)[0]
            if first_query_word in name_words:
                score += 5
        
        # 4. Độ dài description (ngắn hơn = tổng quát hơn = ưu tiên thấp hơn)
        # Code dài hơn thường chi tiết hơn
        score += len(item['code']) * 0.1
        
        if score > 0:
            candidates.append({
                'code': item['code'],
                'name': item['name'],
                'chapter': item['chapter'],
                'score': score
            })
    
    # Sắp xếp theo điểm
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Trả về top candidates
    return candidates[:max_results]

def fetch_caselaw_hierarchy(hs_code):
    if not hs_code:
        return {"chapter": "", "chapter_groups": {}}

    url = f"https://caselaw.vn/ket-qua-tra-cuu-ma-hs?query={quote_plus(str(hs_code))}"
    html_text = None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        html_text = resp.text
    except Exception:
        return {"chapter": "", "chapter_groups": {}}

    soup = BeautifulSoup(html_text, "html.parser")
    chapters = []
    chapter_groups = {}
    current_ch = ""
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    chapter_re = re.compile(r'^(Chương\s+\d+)\s*-?\s*(.*)$', re.I)
    code_re = re.compile(r'^(\d{4,10})\s*-?\s*(.*)$')

    for i, ln in enumerate(lines):
        cm = chapter_re.match(ln)
        if cm:
            title = cm.group(1).strip()
            desc = cm.group(2).strip()
            if not desc and i+1 < len(lines):
                desc_candidate = lines[i+1].strip()
                if not desc_candidate.lower().startswith("chuong "):
                    desc = desc_candidate
            chapter_title = f"{title} – {desc}" if desc else title
            chapters.append(chapter_title)
            current_ch = chapter_title
            chapter_groups.setdefault(current_ch, [])
            continue

        cm2 = code_re.match(ln)
        if cm2 and current_ch:
            code = cm2.group(1)
            tail = cm2.group(2).strip()
            if not tail and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if not next_line.lower().startswith("chuong "):
                    tail = next_line
            label = f"{code} – {tail}" if tail else code
            chapter_groups[current_ch].append(label)

    return {"chapter": chapters[0] if chapters else "", "chapter_groups": chapter_groups}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    user_query = ""
    hs_data_list = []

    if request.method == "POST":
        user_query = request.form.get("description", "").strip()
        if user_query and HS_DATABASE:
            
            # Bước 1: Tìm kiếm thông minh candidates
            candidates = smart_filter_candidates(user_query, max_results=30)
            
            if not candidates:
                print("⚠️ Không tìm thấy candidates phù hợp")
                return render_template("index.html", user_query=user_query, hs_data_list=hs_data_list)
            
            print(f"🔍 Tìm được {len(candidates)} candidates")
            print(f"Top 3: {candidates[:3]}")
            
            # Bước 2: Cho AI chọn code tốt nhất từ danh sách candidates
            candidates_json = json.dumps(
                [{"code": c['code'], "name": c['name']} for c in candidates],
                ensure_ascii=False,
                indent=2
            )
            
            system_prompt = f"""
Bạn là chuyên gia phân loại HS code (Harmonized System Code).

DANH SÁCH HS CODE ỨNG VIÊN (đã lọc sẵn phù hợp):
{candidates_json}

NHIỆM VỤ:
1. Phân tích mô tả sản phẩm từ user
2. Chọn **1 HS code chính xác nhất** từ danh sách trên
3. CHÚ Ý: 
   - Code càng DÀI (8-10 số) = càng CHI TIẾT, PHÙ HỢP với mô tả cụ thể
   - Code càng NGẮN (4-6 số) = càng TỔNG QUÁT
   - Nếu user mô tả chi tiết (có tính chất, đặc điểm cụ thể) → chọn code DÀI
   - Nếu user mô tả chung chung → chọn code NGẮN

Trả về JSON:
{{"code": "96081090", "reason": "lý do chọn code này"}}

QUY TẮC:
- CHỈ chọn code từ danh sách trên
- Trả về JSON hợp lệ
"""
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Mô tả sản phẩm: {user_query}"}
                    ],
                    temperature=0.1,
                    max_completion_tokens=150,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"}
                )
                
                ai_response = completion.choices[0].message.content.strip()
                print(f"🤖 AI Response: {ai_response}")
                
                # Parse JSON response
                data = json.loads(ai_response)
                selected_code = data.get('code', '').strip()
                
                # Tìm code trong candidates
                final_item = None
                for candidate in candidates:
                    if candidate['code'] == selected_code:
                        final_item = candidate
                        break
                
                # Fallback: nếu AI chọn sai, lấy top 1
                if not final_item:
                    print("⚠️ AI chọn code không có trong candidates, fallback top 1")
                    final_item = candidates[0]
                
                # Fetch thông tin từ caselaw
                hs_result = fetch_caselaw_hierarchy(final_item['code'])
                
                hs_data_list.append({
                    "code": final_item['code'],
                    "name": final_item['name'],
                    "chapter": hs_result["chapter"],
                    "groups": [c for c in hs_result["chapter_groups"].get(hs_result["chapter"], []) 
                              if c.startswith(final_item['code'][:4]) or c.startswith(final_item['code'][:6]) or c.startswith(final_item['code'][:8])]
                })
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                # Fallback: lấy top 1 candidate
                if candidates:
                    final_item = candidates[0]
                    hs_result = fetch_caselaw_hierarchy(final_item['code'])
                    hs_data_list.append({
                        "code": final_item['code'],
                        "name": final_item['name'],
                        "chapter": hs_result["chapter"],
                        "groups": []
                    })

    return render_template("index.html", user_query=user_query, hs_data_list=hs_data_list)

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"✓ Đã load {len(HS_DATABASE)} HS codes từ output.json")
    if HS_DATABASE:
        print(f"✓ Code mẫu: {HS_DATABASE[0]['code']} - {HS_DATABASE[0]['name']}")
        # Test tìm kiếm
        test_results = smart_filter_candidates("bút bi", max_results=5)
        print(f"✓ Test search 'bút bi': {len(test_results)} kết quả")
        if test_results:
            print(f"  Top 1: {test_results[0]['code']} - {test_results[0]['name']}")
    else:
        print("❌ Không có dữ liệu! Kiểm tra file output.json")
    print(f"{'='*60}")
    app.run(debug=True)