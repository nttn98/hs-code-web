# version.py
import subprocess
from flask import Blueprint, jsonify

version_bp = Blueprint("version", __name__)

def get_commit_count_from_shortlog():
    try:
        output = subprocess.check_output(
            ["git", "shortlog", "-sn", "--all"],
            stderr=subprocess.STDOUT,
            text=True
        )

        total = 0
        for line in output.strip().splitlines():
            # mỗi dòng: "123 Name"
            count = int(line.strip().split()[0])
            total += count

        return total
    except Exception:
        return None


def get_git_version():
    # Ưu tiên tag nếu có
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.STDOUT,
            text=True
        ).strip()

        commit_count = get_commit_count_from_shortlog()
        if commit_count:
            return f"{tag}.{commit_count}"

        return tag

    except subprocess.CalledProcessError:
        commit_count = get_commit_count_from_shortlog()
        if commit_count:
            return f"v0.1.{commit_count}"

        # fallback cuối
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
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
