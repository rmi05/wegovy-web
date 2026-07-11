from pathlib import Path

from flask import Flask, jsonify, request

from wegobe import compare_predictions

app = Flask(__name__, static_folder='.', static_url_path='')

html = Path('index.html').read_text(encoding='utf-8')


@app.route('/')
def index():
    return html


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')

    result = compare_predictions(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
