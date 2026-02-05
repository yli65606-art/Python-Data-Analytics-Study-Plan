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










