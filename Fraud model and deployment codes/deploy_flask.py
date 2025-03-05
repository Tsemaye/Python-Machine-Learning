from flask import Flask, request, jsonify  # Importing Flask for building the API, request for handling input, and jsonify for returning JSON responses
from flask_cors import CORS  # This allows cross-origin requests (useful for frontend apps and Postman testing)
import joblib  # Used for loading our trained machine learning models
import pandas as pd  # Pandas helps us handle tabular data
import os  # OS module is used for handling file paths

# Initializing Flask app
app = Flask(__name__)  # This starts the Flask application
CORS(app)  # Enable CORS to allow external requests (like from a frontend or Postman)

# The path where our trained models are stored
models_path = "/Users/charisoneyemi/Downloads/611Assignment/Models"

# Loading trained models into a dictionary
# This ensures all models are ready to make predictions when needed
models = {
    "SVM": joblib.load(os.path.join(models_path, "svm_model.pkl")),  # Support Vector Machine
    "KNN": joblib.load(os.path.join(models_path, "knn_model.pkl")),  # K-Nearest Neighbors
    "NaiveBayes": joblib.load(os.path.join(models_path, "nb_model.pkl")),  # Naïve Bayes
    "MLP": joblib.load(os.path.join(models_path, "mlp_model.pkl")),  # Multi-Layer Perceptron
    "XGBoost": joblib.load(os.path.join(models_path, "xgb_model.pkl"))  # XGBoost
}

# Default route to check if the API is running
@app.route("/")
def home():
    return jsonify({"message": "Flask Fraud Detection API is running!"})  # A simple message to confirm the API is live

# This route handles fraud detection predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extracting JSON data from the request
        data = request.get_json()
        # Converting the data into a Pandas DataFrame (Flask expects lists, so we wrap it inside a list)
        df = pd.DataFrame([data["data"]])

        predictions = {}  # Dictionary to store predictions from each model
        for name, model in models.items():  # Loop through each model
            preds = model.predict(df)  # Make a prediction
            predictions[name] = preds.tolist()  # Convert NumPy array to a Python list for JSON response

        return jsonify({"predictions": predictions})  # Return predictions as a JSON response

    except Exception as e:
        # If an error occurs (e.g., incorrect input format), return a 400 error
        return jsonify({"error": str(e)}), 400

# Running the Flask application
if __name__ == "__main__":
    # Flask runs on port 5000 by default, and debug mode is enabled for easier troubleshooting
    app.run(debug=True, host="0.0.0.0", port=5000)  
    # "0.0.0.0" allows access from any device on the network
