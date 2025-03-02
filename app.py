from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import requests
import random

app = Flask(__name__)
CORS(app)
# Load the trained model
model = joblib.load("crop_prediction_model.pkl")

# Function to fetch sensor data and make a prediction
def get_prediction():
    url = "https://blr1.blynk.cloud/external/api/getAll?token=Qwp8at07oadKbKvA1T6p4Wcp8iO6fk4s"
    r = requests.get(url)

    temp = float(r.json().get("v0", 25))  
    humidity = float(r.json().get("v1", 50))
    ph = random.randint(3, 10)
    soil_moisture = float(r.json().get("v3", 30))

    predict_data = [[temp, humidity, ph]]
    predicted_crop = model.predict(predict_data)[0]

    return {"temperature": temp, "humidity": humidity, "ph": ph, "soil_moisture": soil_moisture, "prediction": predicted_crop}

# Define an API endpoint for prediction
@app.route("/predict", methods=["GET"])
def predict():
    try:
        result = get_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
