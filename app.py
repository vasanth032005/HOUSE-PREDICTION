from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])

        # prediction
        prediction = model.predict([[area]])[0]

        # ✅ FIX: prevent negative values
        prediction = max(0, prediction)

        return render_template(
            "index.html",
            prediction_text=f"Price: {prediction:.2f} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error in prediction"
        )

if __name__ == "__main__":
    app.run(debug=True)