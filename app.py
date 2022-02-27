from flask import Flask, request, jsonify
from flask_cors import CORS
from models import startModel, predict
import json

startModel()
app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/api/answer", methods=["POST"])
def answer():
    data = json.loads(request.get_data())
    sent = data['query']

    predictData = predict(sent)
    print(100 * '-')

    return jsonify(predictData)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=88088, threaded=True)
