# Diebetes risk & lifestyle factors combining MySQL & Python

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector # Import the selector
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics
import pickle
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import mysql.connector 

# --- Ensure these variables use your NEW password from the install ---
HOST = "127.0.0.1"          
USER = "root"
PASSWORD = "NewPass123"  # <-- REPLACE with the exact new password you just set
DATABASE = "diabetes_project"
TABLE_NAME = "diabetes-risk-and-lifestyle-factors" 

# --- Connection and Data Loading ---
try:
    engine = create_engine(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}')
    sql_query = f"SELECT * FROM `{TABLE_NAME}`;"
    
    # Loads the SQL data directly into a Pandas DataFrame
    df_from_sql = pd.read_sql_query(sql_query, engine)
    engine.dispose()
    
    print("Successfully connected to MySQL with Python!")
    print(df_from_sql.head())
    
    # --- Place your existing ML code here using df_from_sql ---
    # The rest of your Gradient Boosting model code goes here
    # --------------------------------------------------------

except Exception as e:
    print(f"Error connecting with Python: {e}")
#Create the Preprocessing Pipeline
# Numerical: Fill missing with median and scale the data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Categorical: Fill missing with most frequent and apply One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        # Selects all numerical columns automatically
        ('num', numeric_transformer, make_column_selector(dtype_include=['number'])), 
        # Selects all text/object columns automatically
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object'])) 
    ])
# Modeling
# Gradient boosting model:
# Prepare data for modeling
X = df_from_sql.drop(columns=['Diabetes_Status'])
y = df_from_sql['Diabetes_Status']

#Create and train the full pipline 
from sklearn.ensemble import GradientBoostingClassifier
model_pipeline_gbc = Pipeline(steps=[
    ('preprocessor', preprocessor), # Your existing preprocessor object
    # FIX: Using Classifier with standard 2025 hyperparameters
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])
#Split data and fit the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline_gbc.fit(X_train, y_train)

# predict() gives the 0 or 1 label outcome
y_pred_gbc = model_pipeline_gbc.predict(X_test)
# predict_proba() gives the actual risk score (e.g., 0.85 chance of diabetes)

# Evaluate the model using CLASSIFICATION metrics
accuracy = metrics.accuracy_score(y_test, y_pred_gbc)
conf_matrix = metrics.confusion_matrix(y_test, y_pred_gbc)
class_report = metrics.classification_report(y_test, y_pred_gbc)

print(f"GB Classifier Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualization (Use a better visualization for classification results)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Diabetes Prediction')
plt.show()