import json
import os
import re
from flask import Blueprint, request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
from functools import lru_cache
import logging
from difflib import SequenceMatcher
from hashlib import md5

from groq import Groq

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# ENV
# =====================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_AI = os.getenv("MODEL_AI", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("❌ Thiếu GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
logger.info(f"✓ Using Groq model: {MODEL_AI}")

# =====================================================
# CONFIG
# =====================================================
hs_external_bp = Blueprint("hs_external", __name__)
DATA_FILE = "./data/global.json"

# =====================================================
# LOAD + FLATTEN GLOBAL.JSON (CACHED)
# =====================================================
@lru_cache(maxsize=1)
def load_global_data():
    """Load global data with caching"""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("data", [])
    except FileNotFoundError:
        logger.error(f"File not found: {DATA_FILE}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []

@lru_cache(maxsize=1)
def flatten_hs_tree():
    """
    Flatten hierarchical HS tree into list of leaf codes
    """
    tree = load_global_data()
    rows = []

    for chapter in tree:
        chapter_code = chapter.get("code", "")
        chapter_desc = chapter.get("description", "")

        for leaf in chapter.get("leaf", []):
            group_code = leaf.get("code", "")
            group_desc = leaf.get("name", "")

            for group in leaf.get("group", []):
                if not isinstance(group, dict):
                    continue

                def walk(node, parent_name=""):
                    if not isinstance(node, dict):
                        return

                    code = node.get("code")
                    name = node.get("name", "")
                    
                    # Build full context path
                    full_context = f"{parent_name} > {name}" if parent_name else name

                    # Only add valid HS codes (6-8 digits)
                    if isinstance(code, str) and re.fullmatch(r"\d{6,8}", code):
                        rows.append({
                            "hs_code": code,
                            "hs_name": name,
                            "full_context": full_context,
                            "chapter_code": chapter_code,
                            "chapter_desc": chapter_desc,
                            "group_code": group_code,
                            "group_desc": group_desc,
                        })

                    # Recursively walk children
                    for child in node.get("children", []):
                        walk(child, full_context)

                walk(group)

    logger.info(f"Flattened {len(rows)} HS codes")
    return rows

@lru_cache(maxsize=1)
def get_chapter_summary():
    """
    Create a concise summary of all chapters for AI context
    """
    tree = load_global_data()
    summary = []
    chapters_found = []
    
    for chapter in tree:
        code = chapter.get("code", "")
        desc = chapter.get("description", "")
        if code and desc:
            chapters_found.append(code)
            # Normalize: remove leading zeros for consistency
            normalized_code = code.lstrip("0") or "0"
            summary.append(f"• Chương {normalized_code}: {desc}")
    
    logger.info(f"Loaded {len(chapters_found)} chapters: {chapters_found[:10]}...")
    return "\n".join(summary)

# =====================================================
# SIMPLE CACHE TO REDUCE API CALLS
# =====================================================
@lru_cache(maxsize=100)
def get_cached_chapter(query_hash: str):
    """Cache chapter selection results"""
    return None

# =====================================================
# FALLBACK METHOD
# =====================================================
def simple_chapter_fallback(query: str):
    """
    Fallback: Simple keyword matching when AI is unavailable
    """
    query_lower = query.lower()
    
    # Simple keyword-to-chapter mapping
    keyword_map = {
        "1": ["động vật", "ngựa", "lừa", "trâu", "bò", "lợn", "cừu", "dê", "gà", "vịt", "thỏ"],
        "2": ["thịt", "cá", "hải sản", "tôm", "mực"],
        "3": ["sữa", "trứng", "mật ong"],
        "4": ["cà phê", "chè", "trà"],
        "7": ["rau", "củ", "quả"],
        "8": ["hoa quả", "táo", "chuối", "cam", "thanh long", "xoài", "dưa"],
        "10": ["ngũ cốc", "lúa", "gạo"],
        "84": ["máy móc", "động cơ", "máy bơm"],
        "85": ["điện", "máy tính", "điện thoại"],
        "87": ["xe", "ô tô", "xe máy"],
    }
    
    best_chapter = None
    best_score = 0
    
    for chapter, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in query_lower:
                score = len(keyword)
                if score > best_score:
                    best_score = score
                    best_chapter = chapter
    
    if best_chapter:
        logger.info(f"Fallback selected chapter: {best_chapter} (score: {best_score})")
        return best_chapter
    
    return None

# =====================================================
# GET LEAF GROUPS (4-DIGIT CODES)
# =====================================================
@lru_cache(maxsize=1)
def get_leaf_groups():
    """
    Get all 4-digit leaf groups with their descriptions
    Returns: list of {code, name, chapter_code, chapter_desc}
    """
    tree = load_global_data()
    leaf_groups = []
    
    for chapter in tree:
        chapter_code = chapter.get("code", "")
        chapter_desc = chapter.get("description", "")
        
        for leaf in chapter.get("leaf", []):
            leaf_code = leaf.get("code", "")
            leaf_name = leaf.get("name", "")
            
            if leaf_code and leaf_name:
                leaf_groups.append({
                    "code": leaf_code,
                    "name": leaf_name,
                    "chapter_code": chapter_code,
                    "chapter_desc": chapter_desc
                })
    
    logger.info(f"Loaded {len(leaf_groups)} leaf groups")
    return leaf_groups

# =====================================================
# AI-POWERED HS CODE SELECTION (THREE-STAGE) - GROQ
# =====================================================
def ai_select_chapter(query: str):
    """
    Stage 1: AI selects the most relevant chapter using Groq
    Returns: chapter_code or None
    """
    # Check cache first
    query_hash = md5(query.lower().strip().encode()).hexdigest()
    
    chapter_summary = get_chapter_summary()
    
    prompt = f"""Bạn là chuyên gia phân loại HS Code. Nhiệm vụ: xác định CHƯƠNG HS phù hợp với hàng hóa.

# DANH SÁCH CHƯƠNG HS
{chapter_summary}

# HÀNG HÓA
"{query}"

# YÊU CẦU
Phân tích đặc điểm hàng hóa (vật liệu, công dụng, nguồn gốc, trạng thái) và chọn CHƯƠNG HS phù hợp nhất.

# OUTPUT (CHỈ TRẢ VỀ JSON)
{{
  "chapter": "số chương (VÍ DỤ: 1, 8, 10, 84 - KHÔNG CÓ SỐ 0 ĐẰNG TRƯỚC)",
  "reasoning": "Lý do ngắn gọn (1 câu)"
}}

LƯU Ý: 
- CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC
- Số chương KHÔNG có số 0 đằng trước (ví dụ: "1" chứ không phải "01")"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_AI,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia phân loại HS Code. Chỉ trả về JSON, không có text khác."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            chapter = result.get("chapter", "").strip()
            reasoning = result.get("reasoning", "")
            
            # Normalize chapter: remove leading zeros
            chapter = chapter.lstrip("0") or "0"
            
            logger.info(f"AI selected chapter: {chapter} - {reasoning}")
            return chapter
            
        except Exception as e:
            error_str = str(e)
            
            # Handle rate limit
            if "rate_limit" in error_str.lower() or "429" in error_str:
                import time
                delay = 5 * (attempt + 1)
                
                logger.warning(f"Rate limit hit. Retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    logger.error("Max retries reached, using fallback")
                    return simple_chapter_fallback(query)
            
            # Other errors
            logger.error(f"Chapter selection error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return simple_chapter_fallback(query)
    
    return simple_chapter_fallback(query)

def ai_select_leaf_group(query: str, chapter: str):
    """
    Stage 2: AI selects the most relevant 4-digit leaf group
    Returns: leaf_code or None
    """
    # Get all leaf groups for this chapter
    all_leaf_groups = get_leaf_groups()
    chapter_normalized = chapter.lstrip("0") or "0"
    
    chapter_leaves = [
        leaf for leaf in all_leaf_groups
        if leaf["chapter_code"].lstrip("0") == chapter_normalized
    ]
    
    if not chapter_leaves:
        logger.error(f"No leaf groups found for chapter {chapter}")
        return None
    
    # Build leaf summary
    leaf_text = ""
    for leaf in chapter_leaves[:50]:  # Limit to avoid token overflow
        leaf_text += f"• {leaf['code']}: {leaf['name']}\n"
    
    prompt = f"""Bạn là chuyên gia phân loại HS Code. Nhiệm vụ: chọn NHÓM HÀNG (4 số) phù hợp nhất.

# DANH SÁCH NHÓM HÀNG (Chương {chapter})
{leaf_text}

# HÀNG HÓA
"{query}"

# YÊU CẦU
Phân tích đặc điểm hàng hóa và chọn NHÓM 4 SỐ phù hợp nhất.

# OUTPUT (CHỈ TRẢ VỀ JSON)
{{
  "leaf_code": "mã 4 số (VÍ DỤ: 0309)",
  "reasoning": "Lý do chọn (1 câu)"
}}

LƯU Ý: CHỈ TRẢ VỀ JSON"""

    try:
        response = client.chat.completions.create(
            model=MODEL_AI,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia HS Code. Chỉ trả JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        leaf_code = result.get("leaf_code", "").strip()
        reasoning = result.get("reasoning", "")
        
        logger.info(f"AI selected leaf: {leaf_code} - {reasoning}")
        return leaf_code
        
    except Exception as e:
        logger.error(f"Leaf selection error: {e}")
        return None

def ai_select_specific_code(query: str, chapter: str, leaf_code: str, candidates: list):
    """
    Stage 2: AI selects specific HS code from filtered candidates using Groq
    """
    # Prepare candidates list with full context
    candidates_text = ""
    for idx, item in enumerate(candidates[:100], 1):  # Limit to 100 for token safety
        context = item.get("full_context", item["hs_name"])
        candidates_text += f"{idx}. {item['hs_code']}: {context}\n"
    
    if not candidates_text:
        return None
    
    prompt = f"""Bạn là chuyên gia phân loại HS Code. Nhiệm vụ: chọn MÃ HS CỤ THỂ NHẤT từ danh sách.

# DANH SÁCH MÃ HS (Chương {chapter})
{candidates_text}

# HÀNG HÓA
"{query}"

# YÊU CẦU
1. Phân tích chi tiết đặc điểm hàng hóa
2. So sánh với từng mã HS trong danh sách
3. Chọn mã HS CỤ THỂ NHẤT (8 số nếu có, 6 số nếu không)
4. Đánh giá độ tin cậy:
   - high: Rất chắc chắn (>90%)
   - medium: Khá chắc chắn (70-90%)
   - low: Không chắc chắn (<70%)

# OUTPUT (CHỈ TRẢ VỀ JSON)
{{
  "selected_code": "mã HS 6-8 số",
  "confidence": "high|medium|low",
  "reasoning": "Giải thích chi tiết tại sao chọn mã này (2-3 câu)",
  "key_features": "Đặc điểm quan trọng của hàng hóa đã xem xét"
}}

LƯU Ý:
- Mã phải TỒN TẠI trong danh sách trên
- Ưu tiên mã CỤ THỂ (8 số) hơn mã CHUNG (6 số)
- Nếu không chắc chắn, đặt confidence là "low"
- CHỈ TRẢ VỀ JSON"""

    try:
        response = client.chat.completions.create(
            model=MODEL_AI,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân loại HS Code. Chỉ trả về JSON, không có text khác."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.15,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        selected_code = result.get("selected_code")
        confidence = result.get("confidence", "unknown").lower()
        reasoning = result.get("reasoning", "")
        key_features = result.get("key_features", "")
        
        logger.info(f"AI selected code: {selected_code} (confidence: {confidence})")
        logger.info(f"Reasoning: {reasoning}")
        
        if not selected_code:
            return None
            
        # Find full HS data
        for item in candidates:
            if item["hs_code"] == selected_code:
                return {
                    **item,
                    "ai_confidence": confidence,
                    "ai_reasoning": reasoning,
                    "ai_key_features": key_features
                }
        
        logger.warning(f"Selected code {selected_code} not found in candidates")
        return None
        
    except Exception as e:
        logger.error(f"Code selection error: {e}")
        return None

# =====================================================
# MAIN AI SELECTION PIPELINE
# =====================================================
def ai_select_hs_code(query: str):
    """
    Three-stage AI selection:
    1. Select relevant chapter (2 digits)
    2. Select leaf group (4 digits)
    3. Select specific HS code (6-8 digits)
    """
    all_codes = flatten_hs_tree()
    
    if not all_codes:
        logger.error("No HS codes available")
        return None
    
    # Stage 1: Select chapter
    logger.info("Stage 1: Selecting chapter...")
    chapter = ai_select_chapter(query)
    
    if not chapter:
        logger.error("Failed to select chapter")
        return None
    
    # Stage 2: Select leaf group (4 digits)
    logger.info("Stage 2: Selecting leaf group...")
    leaf_code = ai_select_leaf_group(query, chapter)
    
    if not leaf_code:
        logger.error("Failed to select leaf group")
        # Fallback: use all codes from chapter
        chapter_normalized = chapter.lstrip("0") or "0"
        chapter_codes = [
            item for item in all_codes 
            if item["chapter_code"].lstrip("0") == chapter_normalized
        ]
        
        if not chapter_codes:
            return None
            
        logger.info(f"Fallback: Using all {len(chapter_codes)} codes from chapter")
        selected = ai_select_specific_code(query, chapter, "all", chapter_codes)
        return selected
    
    # Filter codes by chapter AND leaf group (first 4 digits)
    chapter_normalized = chapter.lstrip("0") or "0"
    leaf_codes = [
        item for item in all_codes 
        if item["chapter_code"].lstrip("0") == chapter_normalized
        and item["group_code"].startswith(leaf_code)
    ]
    
    logger.info(f"Found {len(leaf_codes)} codes in chapter {chapter}, leaf {leaf_code}")
    
    if not leaf_codes:
        logger.warning(f"No codes found for leaf {leaf_code}, trying all codes in chapter")
        # Fallback: use all codes from chapter
        chapter_codes = [
            item for item in all_codes 
            if item["chapter_code"].lstrip("0") == chapter_normalized
        ]
        
        if not chapter_codes:
            return None
            
        selected = ai_select_specific_code(query, chapter, leaf_code, chapter_codes)
        return selected
    
    # Stage 3: Select specific code
    logger.info("Stage 3: Selecting specific code...")
    selected = ai_select_specific_code(query, chapter, leaf_code, leaf_codes)
    
    return selected

# =====================================================
# STREAM AI EXPLANATION
# =====================================================
def stream_ai_explain(query: str, hs: dict):
    """
    Stream detailed explanation for selected HS code using Groq
    """
    confidence = hs.get("ai_confidence", "unknown")
    reasoning = hs.get("ai_reasoning", "")
    key_features = hs.get("ai_key_features", "")
    
    confidence_text = {
        "high": "cao ✅",
        "medium": "trung bình ⚠️",
        "low": "thấp ⚠️",
    }.get(confidence, "chưa xác định")

    prompt = f"""Bạn là chuyên gia phân loại HS Code. Hãy giải thích chi tiết kết quả phân loại.

# THÔNG TIN MÃ HS
- Mã HS: {hs['hs_code']}
- Tên: {hs['hs_name']}
- Chương {hs['chapter_code']}: {hs['chapter_desc']}
- Nhóm {hs['group_code']}: {hs['group_desc']}
- Độ tin cậy: {confidence_text}

# PHÂN TÍCH CỦA AI
- Lý do chọn: {reasoning}
- Đặc điểm quan trọng: {key_features}

# HÀNG HÓA
"{query}"

# YÊU CẦU TRÌNH BÀY

## 1. Xác nhận kết quả (ngắn gọn 1-2 câu)
- Khẳng định mã HS này phù hợp/có thể phù hợp
- Giải thích ngắn gọn căn cứ phân loại

## 2. Phân tích chi tiết (ngắn gọn 1-2 điểm)
- Đặc điểm chính của hàng hóa phù hợp với mã HS
- Tiêu chí phân loại quan trọng đã áp dụng
- Phạm vi và giới hạn của mã HS này

## 3. Lưu ý thực tế
- Điều kiện hoặc yêu cầu đặc biệt (nếu có)
- Các mã HS tương tự cần phân biệt (nếu có)
- Khuyến nghị xác minh với cơ quan hải quan
{f"- **⚠️ QUAN TRỌNG**: Độ tin cậy {confidence_text} - Nên tham khảo thêm ý kiến chuyên gia hải quan" if confidence in ["medium", "low"] else ""}

# QUY TẮC
- Sử dụng markdown: **bold**, bullet points, ## headers
- Ngắn gọn, rõ ràng, chuyên nghiệp
- Chỉ dựa vào thông tin đã cung cấp
- KHÔNG suy đoán hoặc thêm thông tin không có
"""

    try:
        # Send metadata first
        metadata = {
            "type": "metadata",
            "data": {
                **hs,
                "confidence": confidence,
                "confidence_text": confidence_text,
                "reasoning": reasoning,
                "key_features": key_features
            }
        }
        yield f"data:{json.dumps(metadata, ensure_ascii=False)}\n\n"

        # Stream explanation
        stream = client.chat.completions.create(
            model=MODEL_AI,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân loại HS Code. Trả lời bằng markdown format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text_data = {
                    "type": "text",
                    "content": chunk.choices[0].delta.content
                }
                yield f"data:{json.dumps(text_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Explanation streaming error: {e}")
        error_data = {
            "type": "error",
            "message": "Lỗi khi tạo giải thích"
        }
        yield f"data:{json.dumps(error_data, ensure_ascii=False)}\n\n"

# =====================================================
# API ENDPOINT
# =====================================================
@hs_external_bp.route("/hs_external", methods=["POST"])
def hs_external():
    """
    Main API endpoint - Two-stage AI classification using Groq
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "message": "Content-Type phải là application/json"
        }), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "message": "JSON không hợp lệ"
        }), 400

    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({
            "success": False,
            "message": "Thiếu trường 'query'"
        }), 400

    if len(query) < 5:
        return jsonify({
            "success": False,
            "message": "Mô tả quá ngắn. Vui lòng mô tả chi tiết hơn (ít nhất 5 ký tự)."
        }), 400

    # AI selection pipeline
    logger.info(f"Processing query: {query}")
    
    try:
        selected_hs = ai_select_hs_code(query)
    except Exception as e:
        logger.error(f"AI selection error: {e}")
        return jsonify({
            "success": False,
            "message": "Lỗi khi phân tích. Vui lòng thử lại."
        }), 500

    if not selected_hs:
        return jsonify({
            "success": True,
            "message": "AI không thể xác định mã HS phù hợp. Đề xuất:\n• Mô tả chi tiết hơn về vật liệu, công dụng\n• Nêu rõ trạng thái (nguyên liệu/thành phẩm)\n• Tham khảo chuyên gia hải quan"
        }), 200

    # Stream explanation
    return Response(
        stream_with_context(stream_ai_explain(query, selected_hs)),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

# =====================================================
# HEALTH CHECK
# =====================================================
@hs_external_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        codes_count = len(flatten_hs_tree())
        return jsonify({
            "status": "healthy",
            "hs_codes_loaded": codes_count,
            "ai_provider": "Groq",
            "model": MODEL_AI
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500