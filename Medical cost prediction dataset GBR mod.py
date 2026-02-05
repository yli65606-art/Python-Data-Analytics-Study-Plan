#Medical cost prediction dataset GBR model

#download and load the datasets
import pandas as pd
import os

from sklearn import metrics
from sklearn.linear_model import LinearRegression
import kagglehub
path = kagglehub.dataset_download("miadul/medical-cost-predication-dataset")

print("Path to dataset files:", path)

csv_file_path = os.path.join(path, 'medical_cost_predication_dataset.csv')
df = pd.read_csv(csv_file_path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor


df['physical_activity_level'] = df['physical_activity_level'].map({'Low' : 0, 'Medium' : 1, 'High' : 2})

# 1. Define your Features (X) and Target (y)
# We exclude 'previous_year_cost' if you want to predict based on health only, 
# but including it usually makes the model much more accurate.
X = df.drop(columns=['annual_medical_cost'])
y = df['annual_medical_cost']

# 2. Identify column types for the pipeline
numeric_features = ['age', 'bmi', 'diabetes', 'hypertension', 'heart_disease', 
                    'asthma',  'stress_level', 
                    'doctor_visits_per_year', 'hospital_admissions', 
                    'medication_count', 'insurance_coverage_pct', 'previous_year_cost']

categorical_features = ['gender', 'smoker', 'insurance_type', 'city_type', 'physical_activity_level',]

# 3. Create the Preprocessing Pipeline
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
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model_pipeline_gbr = Pipeline(steps=[
    ('preprocessor', preprocessor), # Uses your existing preprocessor
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Fit the new model
model_pipeline_gbr.fit(X_train, y_train)

# 3. Make Predictions
y_pred_gbr = model_pipeline_gbr.predict(X_test)

# 4. Evaluate and Compare
print(f"GBR R-squared Score: {metrics.r2_score(y_test, y_pred_gbr):.4f}")
print(f"GBR Mean Absolute Error: ${metrics.mean_absolute_error(y_test, y_pred_gbr):.2f}")

import pickle
import os

# Define a path to save the model file (e.g., in your D:/Github folder)
model_path = 'D:/Github/gbr_medical_cost_model.pkl'

# Save the complete pipeline object (model_pipeline_gbr)
with open(model_path, 'wb') as file:
    pickle.dump(model_pipeline_gbr, file)
    
print(f"Model successfully saved to: {model_path}")

