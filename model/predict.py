import pickle
from flask import Flask, request, jsonify
import pandas as pd

model_file = 'model_C=10.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('german_credit')
@app.route('/predict', methods=['POST'])

def predict():
    user_input = request.get_json()

    X = dv.transform([user_input])
    y_pred = model.predict_proba(X)[0, 1]
    risk = y_pred >= 0.5

    result = {
        'credit risk': float(y_pred),
        'risk': bool(risk)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)