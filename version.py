# version.py
import os
import subprocess
from flask import Blueprint, jsonify

version_bp = Blueprint("version", __name__)

# ================= HELPER =================

def is_vercel():
    """
    Chỉ coi là Vercel khi chạy thật trên Vercel
    (VERCEL=1 do Vercel tự set)
    """
    return os.getenv("VERCEL") == "1"


def run_git(cmd):
    return subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        text=True
    ).strip()


def get_commit_count_from_shortlog():
    """
    CHỈ dùng cho LOCAL
    (CI / Vercel shallow clone -> không tin được)
    """
    try:
        output = run_git(["git", "shortlog", "-sn", "--all"])
        total = 0
        for line in output.splitlines():
            total += int(line.split()[0])
        return total
    except Exception:
        return None


# ================= VERSION LOGIC =================

def get_git_version():
    # ========== VERCEL ==========
    if is_vercel():
        sha = os.getenv("VERCEL_GIT_COMMIT_SHA")
        ref = os.getenv("VERCEL_GIT_COMMIT_REF")

        if sha:
            return f"vercel-{ref}-{sha[:7]}" if ref else f"vercel-{sha[:7]}"

        return "vercel-unknown"

    # ========== LOCAL / NON-VERCEL ==========
    try:
        tag = run_git(["git", "describe", "--tags", "--abbrev=0"])
        commit = run_git(["git", "rev-parse", "--short", "HEAD"])
        count = get_commit_count_from_shortlog()

        if count:
            return f"{tag}.{count}+{commit}"

        return f"{tag}+{commit}"

    except Exception:
        # fallback local
        try:
            commit = run_git(["git", "rev-parse", "--short", "HEAD"])
            return f"dev-local-{commit}"
        except Exception:
            return "dev-local"


# ================= API =================

@version_bp.route("/api/version", methods=["GET"])
def api_version():
    return jsonify({
        "version": get_git_version(),
        "env": "vercel" if is_vercel() else "local",
        "status": "success"
    })
