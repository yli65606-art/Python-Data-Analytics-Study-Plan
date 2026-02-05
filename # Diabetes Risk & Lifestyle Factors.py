# Diabetes Risk & Lifestyle Factors

#download and load the datasets
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
# Download latest version
path = kagglehub.dataset_download("miadul/diabetes-risk-and-lifestyle-factors")

print("Path to dataset files:", path)

csv_file_path = os.path.join(path, 'diabetes-risk-and-lifestyle-factors.csv')
df = pd.read_csv(csv_file_path)

#Inspect column data type and missing values
df.head()
#numerical: Glucose, BMI, Insulin, Age, Stress_Score, Sleep_Hours, Diabetes_status
#catagorical: Gender, Diet_Type, Exercise_Frequency, Heredity, Smoking, Alcohol
df.info()
df.describe()

#Data cleaning: checking for NULL values
df.isnull().sum()

#Missing value strategy: Numeric - median, Categorical - mode.
numeric_columns = [
    'Glucose','BMI','Insulin','Age', 'Stress_Score', 'Sleep_Hours', 'Diabetes_Status']
categorical_columns = ['Gender', 'Diet_Type', 'Exercise_Frequency', 'Heredity', 'Smoking', 'Alcohol']

for n_column_name in numeric_columns:
    df[n_column_name] = df[n_column_name].fillna(df[n_column_name].median())

for c_column_name in categorical_columns:
    df[c_column_name] = df[c_column_name].fillna(df[c_column_name].mode()[0])
print(df.isnull().sum())

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

#Exploratory Data Analysis (EDA)
#Numeric: histogram + KDE (Distribution histogram)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='dodgerblue')
    axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')

plt.tight_layout()
plt.show()

#Categoric: countplot vs target
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    sns.countplot(data=df, x=col, hue='Diabetes_Status', ax=axes[i], palette='Set2')
    axes[i].set_title(f"{'Diebetes_Status'} by {col.replace('_', ' ').title()}")
    axes[i].set_ylabel('Count of Participants')

plt.tight_layout()
plt.show()

#Correlation heatmap
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
# Add titles and labels
plt.title("Correlation Heatmap: Diebetes Status & Risk Factors\n", fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

# Modeling
# Gradient boosting model:
# Prepare data for modeling
X = df.drop(columns=['Diabetes_Status'])
y = df['Diabetes_Status']

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

import pickle
import os

# Define a path to save the model file (e.g., in your D:/Github folder)
model_path = 'D:/Github/gbc_Diabetes_Risk_Factors.pkl'

# Save the complete pipeline object (model_pipeline_gbr)
with open(model_path, 'wb') as file:
    pickle.dump(model_pipeline_gbc, file)
    
print(f"Model successfully saved to: {model_path}")