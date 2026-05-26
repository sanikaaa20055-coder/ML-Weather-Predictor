from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# LOAD MODEL

model = pickle.load(
    open("model.pkl", "rb")
)

# HOME PAGE

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = ""

    if request.method == "POST":

        temperature = float(
            request.form["temperature"]
        )

        humidity = float(
            request.form["humidity"]
        )

        wind_speed = float(
            request.form["wind_speed"]
        )

        pressure = float(
            request.form["pressure"]
        )

        # CREATE DATAFRAME

        data = pd.DataFrame([[
            temperature,
            humidity,
            wind_speed,
            pressure
        ]], columns=[

            "Temp_C",
            "Rel Hum_%",
            "Wind Speed_km/h",
            "Press_kPa"
        ])

        # PREDICT

        result = model.predict(data)[0]

        prediction = result

    return render_template(
        "index.html",
        prediction=prediction
    )

# RUN APP

if __name__ == "__main__":

    app.run(debug=True)
