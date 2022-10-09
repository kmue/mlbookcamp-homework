import pickle
from flask import Flask
from flask import request
from flask import jsonify

def load(filename):
    with open(filename, 'rb') as f_in:
         return pickle.load(f_in)

model = load('model2.bin')
dv = load('dv.bin')


app = Flask('client')

@app.route('/predict', methods = ['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    client = y_pred >= 0.5

    result = {
        'client_proba': float(y_pred),
        'client': bool(client) 
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
