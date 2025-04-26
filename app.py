from flask import Flask, request, jsonify
from flask_cors import CORS
from fetch_live import fetch_live_threads
from similarity import find_similar_threads
import traceback
import os 

app = Flask(__name__)
CORS(app)

@app.route('/live', methods=['POST'])
def live_thread_api():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = fetch_live_threads(query)
        return jsonify({"results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Live fetch failed: {str(e)}"}), 500

@app.route('/search', methods=['POST'])
def semantic_thread_api():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = find_similar_threads(query)
        return jsonify({"results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Semantic search failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host="0.0.0.0", port=port)
