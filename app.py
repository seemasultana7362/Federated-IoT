from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from simulator import run_experiment

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "FedViz Lab API", "mode": "synthetic-demo"})

@app.post("/api/experiment")
def experiment():
    try:
        return jsonify(run_experiment(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5173, debug=True)
