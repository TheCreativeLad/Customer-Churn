# %%
import pandas as pd
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

# %%
data = pd.read_csv('E Commerce Dataset.csv')
data

# %%
data.describe(include='all')

# %%
"""
#### Dropping 'CustomerID' from numerical columns as it's an identifier, not a feature
"""

# %%
data = data.drop("CustomerID", axis=1)

# %%
y = data['Churn']
X = data.drop('Churn', axis=1)

# %%
"""
### Identify categorical and numerical features
"""

# %%
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# %%
"""
## Data Splitting (Training and Test set)
"""

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
"""
### Creating preprocessing pipelines for numerical and categorical data
"""

# %%
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# %%
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %%
"""
### Creating a preprocessor using ColumnTransformer
"""

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %%
"""
### Creating the full machine learning pipeline with SMOTE and the model
"""

# %%
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# %%
"""
### Defining the Hyperparameter Grid
"""

# %%
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
}

# %%
"""
### Performing GridSearchCV
"""

# %%
# cv=5 means 5-fold cross-validation
# n_jobs=-1 means use all available CPU cores to speed up the process
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')


# Fitting the grid search to the data
# This will take some time as it trains a model for every combination of parameters
grid_search.fit(X_train, y_train)

# %%
"""
### Analyzing the results
"""

# %%
print("Best Parameters found: ", grid_search.best_params_)
print("Best F1 Score: ", grid_search.best_score_)

# %%
"""
### Saving the new optimized model
"""

# %%
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'gradient_boosting_model_pipeline.joblib')

# %%
print("\nNew optimized model saved successfully as 'gradient_boosting_model_pipeline.joblib'")

# %%
"""
### Load the newly saved optimized model
"""

# %%
optimized_model = joblib.load('gradient_boosting_model_pipeline.joblib')

# %%
y_pred = optimized_model.predict(X_test)

# %%
"""
### Evaluationg the optimized model
"""

# %%
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# %%
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# %%
"""
### Getting the feature importances from the trained model
"""

# %%
print("\n--- Feature Importances ---")
# Access the feature names and importances from the pipeline
onehot_feature_names = optimized_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

feature_names = numerical_features.tolist() + list(onehot_feature_names)

importances = optimized_model.named_steps['classifier'].feature_importances_

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
print(feature_importance_df.to_markdown(index=False))

# %%
pip freeze > requirements.txt

# %%
data.columns

# %%
conda install -c defaults -c conda-forge ipynb-py-convert

# %%
