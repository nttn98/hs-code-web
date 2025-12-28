from flask import Flask, render_template, request, jsonify, Response
from services.internal.hs_internal import search_similar_products
from services.internal.hs_policy_ai import ask_hs_policy
from version import version_bp  

app = Flask(__name__)

# ================= CHAT MEMORY (INTERNAL TOOL) =================
# memory theo HS code (đủ dùng cho internal)
CHAT_MEMORY = {}

# register blueprint
app.register_blueprint(version_bp)

# ================= PAGES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hs_internal")
def hs_internal_page():
    return render_template("hs_internal.html")

@app.route("/hs_external")
def hs_external_page():
    return render_template("hs_external.html")

# ================= API SEARCH =================
@app.route("/api/hs/internal/similar", methods=["POST"])
def api_hs_internal_similar():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    return jsonify(search_similar_products(query, limit=10))

# ================= API CHAT =================
@app.route("/api/hs/internal/chat", methods=["POST"])
def api_hs_internal_chat():
    data = request.get_json(force=True)
    hs_code = data.get("hs_code")
    question = (data.get("question") or "").strip()

    if not hs_code or not question:
        return jsonify({"error": "Thiếu hs_code hoặc question"}), 400

    history = CHAT_MEMORY.setdefault(hs_code, [])

    def stream():
        for chunk in ask_hs_policy(hs_code, question, history):
            yield chunk

    return Response(stream(), content_type="text/plain")

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
