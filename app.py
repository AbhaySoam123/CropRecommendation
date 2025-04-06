from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form[f]) for f in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    pred = model.predict([data])[0]
    crop = le.inverse_transform([pred])[0]
    return render_template('index.html', prediction=crop)

if __name__ == '__main__':
    app.run(debug=True)
