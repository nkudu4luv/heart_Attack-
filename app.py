from flask import Flask, request, render_template
import joblib
import numpy as np

import os

port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)
app = Flask(__name__)

# Load model and selected features
model = joblib.load("logistic_model.pkl")
features = joblib.load("feature_columns.pkl")

# Descriptions for form
feature_descriptions = {
    "cp": "Chest Pain Type (0–3; higher = typical angina)",
    "thalach": "Max Heart Rate Achieved (0–205 bpm)",
    "oldpeak": "ST Depression (0–7)",
    "ca": "Major Vessels Colored by Fluoroscopy (0–3)",
    "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible)",
    "age": "Age (25–80 years)"
}

# Accepted value ranges
feature_ranges = {
    "cp": (0, 3),
    "thalach": (0, 205),
    "oldpeak": (0, 7),
    "ca": (0, 3),
    "thal": (1, 3),
    "age": (25, 80)
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []

    if request.method == "POST":
        inputs = {}
        try:
            for feat in features:
                val = float(request.form[feat])
                min_val, max_val = feature_ranges[feat]
                if not (min_val <= val <= max_val):
                    errors.append(f"{feat.upper()} must be between {min_val} and {max_val}.")
                inputs[feat] = val

            if not errors:
                input_array = np.array([inputs[feat] for feat in features]).reshape(1, -1)
                prediction = model.predict(input_array)[0]

        except ValueError:
            errors.append("All fields must be numeric.")

    return render_template(
        "index.html",
        features=features,
        descriptions=feature_descriptions,
        prediction=prediction,
        errors=errors,
        feature_limits=feature_ranges
    )

if __name__ == "__main__":
    app.run(debug=True)
