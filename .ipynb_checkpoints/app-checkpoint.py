from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home endpoint
@app.route("/")
def home():
    return "Iris ML API Running Successfully!"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get JSON data
        data = request.get_json()

        # Extract features
        features = data["features"]

        # Convert to array
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)

        # Class names
        class_names = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        result = class_names[prediction[0]]

        return jsonify({
            "prediction": result
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# Run app
if __name__ == "__main__":
    app.run(debug=True)