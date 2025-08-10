from flask import Flask, request
from datetime import datetime
import threading
import os

app = Flask(__name__)
LOG_FILE = os.path.abspath("exfil.log")

log_lock = threading.Lock()


def append_log(line: str) -> None:
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


@app.route("/health")
def health() -> str:
    return "OK"


@app.route("/signal", methods=["POST"])
def signal() -> str:
    marker = request.form.get("marker", "")
    ts = datetime.utcnow().isoformat()
    append_log(f"{ts} SIGNAL marker={marker}")
    return "OK"


@app.route("/logs")
def logs() -> str:
    if not os.path.exists(LOG_FILE):
        return "<pre>(no logs)</pre>"
    with open(LOG_FILE, "r") as f:
        return "<pre>" + f.read() + "</pre>"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
