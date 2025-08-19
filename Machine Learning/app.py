import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib



# 1. Load the trained model pipeline
try:
    model_pipeline = joblib.load('gradient_boosting_model_pipeline.joblib')
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: 'optimized_churn_model_pipeline.joblib' not found. Please make sure the file is in the same directory.")
    exit()

# 2. Initialize the Flask application
app = Flask(__name__)

# 3. Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)

    # Convert the JSON data to a Pandas DataFrame
    # The columns must be in the same order as your training data
    df = pd.DataFrame([data])

    # Make a prediction using the loaded model pipeline
    prediction_result = model_pipeline.predict(df)
    
    # Get the prediction value
    prediction = prediction_result[0]

    # Return the prediction as a JSON response
    if prediction == 1:
        response = {'prediction': 'Churn: Yes'}
    else:
        response = {'prediction': 'Churn: No'}
    
    return jsonify(response)

# 4. Run the Flask app

is_development = os.environ.get('FLASK_ENV') == 'development'

if __name__ == '__main__':
    # The host '0.0.0.0' makes the app accessible externally,
    # and debug=True allows for automatic reloading on code changes
    app.run(host='0.0.0.0', port=5000, debug=is_development)