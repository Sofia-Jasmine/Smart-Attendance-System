import sys
import os
from flask_cors import CORS
from flask import Flask, request, jsonify

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from ml_model.inference import predict_proxy


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predict_proxy(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Smart Attendance System API is running"

if __name__ == "__main__":
    app.run(debug=True)
