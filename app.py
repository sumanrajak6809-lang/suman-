# ===============================
# AQI PREDICTION FLASK APP
# ===============================

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load Saved Model
model = joblib.load(os.path.join(script_dir, "model.pkl"))
encoder = joblib.load(os.path.join(script_dir, "encoder.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    city = request.form["city"]
    date = request.form["date"]
    pm2_5 = float(request.form["pm2_5"])
    pm10 = float(request.form["pm10"])
    no = float(request.form["no"])
    no2 = float(request.form["no2"])
    nox = float(request.form["nox"])
    nh3 = float(request.form["nh3"])
    co = float(request.form["co"])
    so2 = float(request.form["so2"])
    o3 = float(request.form["o3"])
    benzene = float(request.form["benzene"])
    toluene = float(request.form["toluene"])

    # Create DataFrame
    df = pd.DataFrame([{
        "date": date,
        "pm2.5": pm2_5,
        "pm10": pm10,
        "no": no,
        "no2": no2,
        "nox": nox,
        "nh3": nh3,
        "co": co,
        "so2": so2,
        "o3": o3,
        "benzene": benzene,
        "toluene": toluene
    }])

    # Date Feature Engineering
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    df.drop("date", axis=1, inplace=True)

    # Prediction
    prediction = model.predict(df)
    result = encoder.inverse_transform(prediction)[0]

    return render_template("index.html", prediction_text=f"Predicted AQI Category: {result}")


if __name__ == "__main__":
    app.run(debug=True)