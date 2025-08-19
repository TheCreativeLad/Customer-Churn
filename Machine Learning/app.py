from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the entire trained pipeline, which includes the preprocessor and the classifier.
try:
    model_pipeline = joblib.load('gradient_boosting_model_pipeline.joblib')
except FileNotFoundError:
    print("Error: The model file 'gradient_boosting_model_pipeline.joblib' was not found.")
    print("Please ensure the model file is in the same directory as this script.")
    exit()

# Extract the list of original numerical and categorical features from your notebook.
# We will use this to manually ensure the order of features is correct.
NUMERICAL_FEATURES = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                      'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear', 
                      'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
                      
CATEGORICAL_FEATURES = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 
                        'PreferedOrderCat', 'MaritalStatus', 'Complain']

# Initialize the Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON data via a POST request, processes it, and returns a churn prediction.
    """
    try:
        # Get the JSON data from the request body
        json_data = request.get_json(force=True)
        
        # Convert the JSON data into a pandas DataFrame.
        # It's crucial that this DataFrame is built from the json data.
        df = pd.DataFrame([json_data])
        
        # --- CRITICAL STEP: Manually reorder columns to match the model's expectations ---
        # We combine the numerical and categorical feature lists in the correct order.
        expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        
        # We reindex the DataFrame to match the expected order. This ensures consistency
        # regardless of how the JSON data was sent.
        # We also fill missing values with 0 as a safe fallback.
        df_reordered = df.reindex(columns=expected_features, fill_value=0)
        
        # Make a prediction using the reordered DataFrame.
        # The model_pipeline will handle all the imputation, scaling, and one-hot encoding automatically.
        prediction = model_pipeline.predict(df_reordered)
        
        # Convert the numpy array prediction to a string and return it
        result = "Yes" if prediction[0] == 1 else "No"
        
        return jsonify({"prediction": result})
        
    except Exception as e:
        # Return a clear error message if something goes wrong.
        return jsonify({"error": str(e), "message": "An error occurred during prediction. Please check your JSON format."}), 400

if __name__ == '__main__':
    # Start the Flask application
    # Set host='0.0.0.0' to make the server externally visible
    app.run(host='0.0.0.0', port=5000, debug=True)
