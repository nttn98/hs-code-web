from flask import Flask, render_template
from version import version_bp
from services.external.hs_external import hs_external_bp
from services.internal.hs_internal import hs_internal_bp   

app = Flask(__name__)

# ================= REGISTER BLUEPRINTS =================
app.register_blueprint(version_bp)
app.register_blueprint(hs_external_bp, url_prefix='/api/hs/external')
app.register_blueprint(hs_internal_bp)   

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

if __name__ == '__main__':
    print("Server đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
