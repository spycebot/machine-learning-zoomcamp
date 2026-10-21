from flask import Flask
from datetime import datetime

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return f"PONG {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

# python ping.py
# https://studious-space-winner-jvgqr5xpgr2pqgx-5000.app.github.dev/ping
# PONG 2023-01-01 12:00:00