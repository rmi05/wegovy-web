from pathlib import Path

from wegobe import compare_predictions

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')

    result = compare_predictions(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)