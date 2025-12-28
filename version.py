# version.py

import subprocess
from flask import Blueprint, jsonify

version_bp = Blueprint("version", __name__)

def get_git_version():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.STDOUT,
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.STDOUT,
                text=True
            ).strip()
            return f"dev-{commit}"
        except:
            return "v1.0.0"

@version_bp.route("/api/version", methods=["GET"])
def api_version():
    return jsonify({
        "version": get_git_version(),
        "status": "success"
    })
