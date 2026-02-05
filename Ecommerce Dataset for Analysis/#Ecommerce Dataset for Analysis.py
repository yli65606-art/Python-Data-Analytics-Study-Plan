#Ecommerce Dataset for Analysis

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import os

#Data download from Kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nabihazahid/ecommerce-dataset-for-sql-analysis")

print("Path to dataset files:", path)
csv_file_path = os.path.join(path, 'ecommerce_dataset_10000.csv')
df = pd.read_csv(csv_file_path)

# Database connection parameters
HOST = "127.0.0.1"          
USER = "root"
PASSWORD = "NewPass123" 
DATABASE = "ecommerce_dataset_10000"


# 1. Create the Database in MySQL
conn = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
cursor = conn.cursor()
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
conn.close()

# 2. Setup SQLAlchemy Engine for Pandas
engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DATABASE}")

# Load the Kaggle dataset 
df.head()
df.isnull().sum()
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['order_date'] = pd.to_datetime(df['order_date'])
df['review_date'] = pd.to_datetime(df['review_date'])

# 3. Load DataFrame into MySQL Database
df.to_sql(name='ecommerce_data', con=engine, if_exists='replace', index=False)
print("Data successfully loaded to MySQL!")

# 4. Visualization of processed data from MySQL
import matplotlib.pyplot as plt
import seaborn as sns

# 5. Query data from MySQL
view_name = 'View_Monthly_Revenue'
query = f"SELECT * FROM {view_name} ORDER BY Month"

df = pd.read_sql(query, con=engine)
df.head()
df['Month'] = pd.to_datetime(df['Month'])

# 6. Plot Monthly Revenue
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Month', weights='Monthly_Revenue', bins=12, kde=True)
plt.title('Monthly Revenue from E-commerce Dataset')
plt.xlabel('Month')
plt.ylabel('Monthly Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Query data from MySQL
view_name = 'view_customer_segments_tiers'
query = f"SELECT * FROM {view_name} ORDER BY customer_id"
df = pd.read_sql(query, con=engine)
df.head()
df['Customer_Segment_Tier'] = df['Customer_Segment_Tier'].map({'Low Value' : 0, 'Mid Value' : 1, 'High Value' : 2})

# 8. Plot Customer Segments
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Customer_Segment_Tier', palette='viridis')
plt.title('Customer Segments Tiers from E-commerce Dataset')
plt.xlabel('Customer Segment Tier (0: Low, 1: Mid, 2: High)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()