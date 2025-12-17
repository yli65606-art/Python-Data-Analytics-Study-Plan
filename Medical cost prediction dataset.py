#Medical cost prediction dataset

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

#Extract info of the loaded dataset
df.head()
df.info()
df.describe()

#Data cleaning: checking for NULL values
df.isnull().sum() #NaN NULL value in Insurance type

#Fill missing insurance_type with the most common value
df['insurance_type'] = df['insurance_type'].fillna(df['insurance_type'].mode()[0])

# Data cleaning: Tranform objects to int
# Binary (yes/no): smoker, gender, insurance_type
df['smoker'] = (df['smoker'] == 'Yes').astype(int)
df['gender'] = (df['gender'] == 'Male').astype(int)
df['insurance_type'] = (df['insurance_type'] == 'Private').astype(int)
# Ordinal (Ranked): physical_activity_level (Low < Medium < High)
df['physical_activity_level'] = df['physical_activity_level'].map({'Low' : 0, 'Medium' : 1, 'High' : 2})
# Nominal (Lables): city_type
df = pd.get_dummies(df, columns=['city_type'], drop_first=True)

df['insurance_type'].unique()
df['city_type'].unique()

#Data visualization (histogram)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

custom_colors = ['red', 'black']
feature_columns = [
    'age',
    'gender',
    'bmi',
    'smoker',
    'diabetes',
    'hypertension',
    'heart_disease',
    'asthma',
    'hospital_admissions',
    'medication_count',
    'insurance_type',
    'insurance_coverage_pct,'
    'previous_year_cost',
    'city_type']
for column_name in feature_columns:
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=df, x='annual_medical_cost', y=column_name)
    plt.title(f"annual_medical_cost vs. {column_name.replace('_', ' ').title()}")
    plt.show()

#Distribution histogram
cols_to_plot = ['age', 'bmi', 'annual_medical_cost', 'previous_year_cost']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='dodgerblue')
    axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')

plt.tight_layout()
plt.show()# annual_medical_cost is right skewed suggesting a small number of high cost

#medical cost vs smoking

import numpy as np # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

#boxplot categorical values vs medical_cost
feature_columns = [
    'smoker',
    'gender',
    'diabetes',
    'hypertension',
    'heart_disease',
    'asthma',
    'hospital_admissions',
    'medication_count',
    'insurance_type',
    'city_type_Semi-Urban',
    'city_type_Urban']

for column_name in feature_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x=column_name, y='annual_medical_cost', palette='Set2')
    plt.title(f"Medical cost vs. {column_name.replace('_', ' ').title()}")
    plt.show()

#continuous numerical values vs medical cost
numerical_features = ['insurance_coverage_pct', 'previous_year_cost', 'medication_count']

for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=col, y='annual_medical_cost', alpha=0.3)
    plt.title(f"Relationship: {col.replace('_', ' ').title()} vs. Cost")
    plt.show()

#Correlation heatmap
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
# Generate the heatmap
# annot=True: Displays the actual correlation numbers in the cells
# cmap='coolwarm': Red for positive correlation, Blue for negative
# vmin/vmax: Sets the color scale range to the standard -1 to 1
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
# Add titles and labels
plt.title("Correlation Heatmap: Medical Factors & Annual Costs\n", fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Filtered Correlation Heatmap (Lower Triangle Only)\n", fontweight='bold', fontsize=18)
plt.show()
#Strong Positive (+0.7 to +1.0): As this factor increases, medical costs go up significantly. You will likely see this for previous_year_cost.
#Moderate Positive (+0.3 to +0.7): Factors like smoker, age, or bmi typically fall here.
#Near Zero (-0.1 to +0.1): These factors have almost no linear relationship with cost and might be less useful for your regression model.
#Negative (-0.1 to -1.0): Rare in medical costs, but this would mean as the factor increases, the cost decreases.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics

# 1. Define your Features (X) and Target (y)
# We exclude 'previous_year_cost' if you want to predict based on health only, 
# but including it usually makes the model much more accurate.
X = df.drop(columns=['annual_medical_cost'])
y = df['annual_medical_cost']

# 2. Identify column types for the pipeline
numeric_features = ['age', 'bmi', 'diabetes', 'hypertension', 'heart_disease', 
                    'asthma', 'physical_activity_level', 'stress_level', 
                    'doctor_visits_per_year', 'hospital_admissions', 
                    'medication_count', 'insurance_coverage_pct', 'previous_year_cost']

categorical_features = ['gender', 'smoker', 'insurance_type', 'city_type_Semi-Urban', 'city_type_Urban']

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

# 4. Create and Train the Full Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 5. Make Predictions and Evaluate
y_pred = model_pipeline.predict(X_test)

print(f"R-squared Score: {metrics.r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error (MAE): ${metrics.mean_absolute_error(y_test, y_pred):.2f}")

#Visualize the model
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Medical Cost: Actual vs. Predicted (R² = 0.91)', fontweight='bold')
plt.xlabel('Actual Annual Cost ($)')
plt.ylabel('Predicted Annual Cost ($)')
plt.show()

#Gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics

# 1. Update the Pipeline with GradientBoostingRegressor
# We use standard 2025 hyperparameters: 100 trees, depth of 4
model_pipeline_gbr = Pipeline(steps=[
    ('preprocessor', preprocessor), # Uses your existing preprocessor
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

# 2. Fit the new model
model_pipeline_gbr.fit(X_train, y_train)

# 3. Make Predictions
y_pred_gbr = model_pipeline_gbr.predict(X_test)

# 4. Evaluate and Compare
print(f"GBR R-squared Score: {metrics.r2_score(y_test, y_pred_gbr):.4f}")
print(f"GBR Mean Absolute Error: ${metrics.mean_absolute_error(y_test, y_pred_gbr):.2f}")

# Extract the importance of each feature from the fitted model
importances = model_pipeline_gbr.named_steps['regressor'].feature_importances_

# (Note: Feature names can be tricky to extract from a Pipeline; 
# for now, this shows the raw importance scores)
print("\nTop Feature Importance Scores:")
print(importances)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_gbr, alpha=0.4, color='crimson')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Gradient Boosting: Actual vs. Predicted (R² = 0.99)', fontweight='bold')
plt.xlabel('Actual Annual Cost ($)')
plt.ylabel('Predicted Annual Cost ($)')
plt.show()
