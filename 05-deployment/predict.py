import pickle
from flask import Flask, jsonify, request
from datetime import datetime

# gunicorn --bind 0.0.0.0:9696 predict:app

app = Flask('churn')

C = 1.0

input_file = f'model_C={C}.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn),
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")