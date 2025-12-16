#Python Fundamentals (Weeks 1-4)
  #Syntax and Basics
    #Variables (letters, numbers, underscores)
    #Data types (A. Integers (int), B. Floating-Point numbers(float), C. Strings(str), D. Booleans(bool))

your_first_name = 'Yusheng'
your_age = 'Thrity three'
your_gpa = 3.5
is_registered = True

print(your_first_name, your_age, your_gpa, is_registered)

A = 3.9
B = int(A)
print('This year is ' + str(B))

math_score = 95
science_score = 88
def average_score(scores):
    return sum(scores) / len(scores)
print(average_score([math_score, science_score]))


dollars = 10
item = "Widget"
print('I have ' + str(dollars) + ' dollars to buy a ' + str(item))
print('I have 10 dollars to buy a Widget')

monthly_sales = [12000, 15000, 11000, 18000]
monthly_sales[2] = 13000
monthly_sales.append(20000)
print(monthly_sales)
data_analyst_skills = ("Python", "SQL", "Visualization")
print(data_analyst_skills[1])

student_info = {
    "name": "Sarah",
    "age": 33,
    "course": "Data Analytics",
    "active_student": True
}

student_name = student_info["name"]
print(student_name)
student_info["age"] = 34

city_data = {
    "city" : "New York",
    "population" : "8399000",
    "country" : "USA"
}
print(city_data["population"])

city_data["population"] = 8400000
city_data["area_sq_miles"] = 302.6
print(city_data)

error_codes = [101, 102, 103, 101, 104, 102, 105, 103]
unique_errors = set(error_codes)
print(unique_errors)

score = 85
if score >= 90:
    print("Grade: A")
elif 89 >= score >= 80:
    print("Grade: B")
elif 79 >= score >= 70:
    print("Grade: C")
elif 69 >= score >= 60:
    print("Grade: D")
else:
    print("Grade: F")  

city_data = {"city": "New York", "population": 8400000}

for key, value in city_data.items():
    print(f"The {key} is {value}.")

# range(5) generates numbers 0, 1, 2, 3, 4
for i in range(5):
    print(f"Loop iteration number: {i}")

monthly_sales = [12000, 15000, 13000, 18000, 20000]
for i in monthly_sales:
    print(f"Sales for this month were {i}")

count = 5
while count > 0:
    print(f"Count is: {count}")
    count = count - 1
if count == 0:
    print("Lift off!")

# 'def' defines the function
# 'greet_user' is the function name
# 'name' is a parameter (input the function expects)
def greet_user(name):
    print(f"Hello, {name}!")
    print("Welcome back.")

# This is 'calling' the function
greet_user("Sarah")

def calculate_total(price, quantity):
    total = price * quantity
    return total # Send the result back to where the function was called

# Call the function and store its returned value in a variable
total = calculate_total(10, 3)
print(total) # Output: 30

# Import the entire math module
import math

# Use a function from the module using the dot notation
circumference = 2 * math.pi * 5
print(circumference)

def calculate_average(scores):
    average = sum(scores)/len(scores)
    return average
test_scores = [99, 100, 95, 93]
final_average = calculate_average(test_scores)
print(final_average)

with open('output.txt', 'w') as file:
    file.write('This is my first line of text.\n')
    file.write('This is my second line of text.\n')

with open('output.txt', 'r') as file:
    content = file.read()
    print(content)

with open('output.txt', 'a') as file:
    file.write('This is the final line.\n')

with open('report.txt', 'w') as file:
    file.write('Project Status Report.\n')
    file.write('Phase 1: Complete.\n')
    file.write('Phase 2: In Progress.\n')

with open('report.txt', 'r') as file:
    content = file.read()
    print(content)
    for line in file:
        print(line.strip())

import numpy as np
data_list = [10, 20, 30, 40, 50]
numpy_array = np.array(data_list)
print(numpy_array)
print(type(numpy_array))

scores = np.array([50, 60, 70, 80])

# Add 10 points to every score instantly (Vectorization!)
adjusted_scores = scores + 10
print(adjusted_scores) # Output:

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Mean (Average): {np.mean(data)}")
print(f"Median (Middle): {np.median(data)}")
print(f"Sum of elements: {np.sum(data)}")
print(f"Max value: {np.max(data)}")

zeros = np.zeros(5)      # Output: [0. 0. 0. 0. 0.]
ones = np.ones(3)        # Output: [1. 1. 1.]
range_array = np.arange(0, 10, 2) # Output: [0 2 4 6 8] (Start, Stop (exclusive), Step)

import numpy as np
temperatures = [72, 75, 78, 60, 65, 70, 73]
numpy_array = np.array(temperatures)
print(numpy_array)

temperatures = np.array([72, 75, 78, 60, 65, 70, 73])
adjusted_temperatures = (temperatures - 32) * 5 / 9
print(adjusted_temperatures)

temperatures = np.array([72, 75, 78, 60, 65, 70, 73])
print(f"Mean (Average): {np.mean(temperatures)}")
print(f"Max value: {np.max(temperatures)}")

import pandas as pd
data = {
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Population_Millions': [8.4, 8.9, 2.1, 13.9],
    'Country': ['USA', 'UK', 'France', 'Japan']
}
# Create the DataFrame
df = pd.DataFrame(data)
print(df)

# View the first 5 rows (or specify a number like df.head(3))
df.head()

# View information about the columns, data types, and missing values
df.info()

# Get quick summary statistics for numerical columns
df.describe()

# Select a single column (returns a Pandas Series object)
populations = df['Population_Millions']
print(populations)

import pandas as pd
#Create a dictionary with data for 5 students:
data = {
    'Name': ['Alice', 'Bob', 'Sam', 'Tim', 'Emmy'],
    'Math_score': [90, 98, 97, 88, 100],
    'Science_score': [93, 86, 81, 76, 100]
}
#Convert this dictionary into a Pandas DataFrame called student_scores_df
student_score_df = pd.DataFrame(data)
print(student_score_df)
#Print the .info() summary of your new DataFrame.
student_score_df.info()
#Select just the Math_Score column and print it.
Score = student_score_df['Math_score']
print(Score)

#Lesson 11: Data Cleaning and Indexing with Pandas
#1. Indexing (Selecting Rows and Columns)
city_name = df.loc[0, 'City']
print(city_name)
all_populations = df.loc[:, 'Population_Millions']
print(all_populations)
third_row_second_col = df.iloc[2, 1]
print(third_row_second_col)
#2. Filtering Data (Boolean Masking)
# The mask is a Series of True/False values
mask = df['Population_Millions'] > 5

print(mask)
# Output:
# 0     True
# 1     True
# 2    False
# 3     True
# Name: Population_Millions, dtype: bool

# Apply the mask to the DataFrame to show only rows where mask is True
wealthy_cities = df[mask]
# or usually done in one line:
wealthy_cities = df[df['Population_Millions'] > 5]

print(wealthy_cities)

#3. Handling Missing Data
# Drop rows with *any* missing values (use with caution, can lose lots of data)
df_cleaned = df.dropna()

# Fill missing values with a specific value (e.g., 0 or the mean of the column)
df_filled = df.fillna(0)

#Filter Data:
high_achievers = student_score_df[(student_score_df['Math_score'] >= 90) & (student_score_df['Science_score'] >= 90)]
print("High Achievers DataFrame:")
print(high_achievers)
#Indexing with .iloc:
third_student = student_score_df.iloc[2, :]
print(third_student)

#Lesson 12: Data Grouping and Aggregation with Pandas

import pandas as pd
data = {
    'Region': ['North', 'North', 'South', 'South', 'North'],
    'Sales': [1000, 1200, 800, 1100, 900],
    'Returns': [50, 60, 40, 55, 45]
}
print(data)
sales_df = pd.DataFrame(data)
print(sales_df)
# Group by the 'Region' column and calculate the average ('mean') for all other numerical columns
regional_summary = sales_df.groupby('Region').mean()
print(regional_summary)

#2. Aggregation Functions (.agg())
summary_advanced = sales_df.groupby('Region').agg({
    'Sales': 'sum',
    'Returns': 'mean'
})
print(summary_advanced)

import pandas as pd
student_data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Math_Score': [90, 98, 85, 92, 78, 88],
    'Science_Score': [93, 86, 89, 95, 80, 91]
}
students_df = pd.DataFrame(student_data)
#Use .groupby() to group the students_df by the Gender column.
#Calculate the average (.mean()) scores for both Math and Science for each gender group.
gender_average_scores = students_df.groupby('Gender').mean()
print(gender_average_scores)
#Using the same grouped data (or grouping again), use .agg() to calculate the maximum ('max') Math_Score and the minimum ('min') Science_Score for each gender.
gender_min_max = students_df.groupby('Gender').agg({
    'Math_Score': 'max',
    'Science_Score': 'min'
})
print(gender_min_max)

#Lesson 13: Data Visualization with Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # You will need pandas for the homework

#2. Basic Plotting with Matplotlib
# Create some simple data
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

# Create the plot area and plot the data
plt.figure(figsize=(8, 5)) # Optional: set the size
plt.plot(x, y)

# Add labels and a title
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.title("Simple Line Plot")

# Display the plot
plt.show()

#3. Using Seaborn with Pandas DataFrames
# Using the student data from the last lesson
student_data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Math_Score': [90, 98, 85, 92, 78, 88],
    'Science_Score': [93, 86, 89, 95, 80, 91]
}
students_df = pd.DataFrame(student_data)

# Create a scatter plot using Seaborn
plt.figure(figsize=(6, 4))
sns.scatterplot(data=students_df, x='Math_Score', y='Science_Score', hue='Gender')
plt.title("Math Score vs Science Score by Gender")
plt.show()

#Create a Bar Plot with Seaborn:
#We want to visualize the average Math score by gender.
#First, calculate the gender_average_scores DataFrame again (the output from your last homework).
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
student_data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Math_Score': [90, 98, 85, 92, 78, 88],
    'Science_Score': [93, 86, 89, 95, 80, 91]
}
students_df = pd.DataFrame(student_data)
gender_average_scores = students_df.groupby('Gender').mean()
print(gender_average_scores)
# Create a scatter plot using Seaborn
plt.figure(figsize=(6, 4))
sns.barplot(data=gender_average_scores, x='Gender', y='Math_Score')
plt.title('gender math scores')
plt.xlabel("Gender")
plt.ylabel("Math Scores")
plt.show()

#Lesson 14: SQL Integration (Connecting Python to Databases)
#2. Using Pandas with SQLite
import pandas as pd
import sqlite3

# 1. Connect to the database
# ':memory:' creates a temporary database just for this session in RAM
conn = sqlite3.connect(':memory:')

# Optional: Create a dummy table in our temporary database so we have data to query
create_table_query = """
CREATE TABLE Employees (
    ID INT,
    Name TEXT,
    Department TEXT,
    Salary INT
);
"""
conn.execute(create_table_query)
conn.execute("INSERT INTO Employees VALUES (1, 'Alice', 'Engineering', 70000)")
conn.execute("INSERT INTO Employees VALUES (2, 'Bob', 'Sales', 60000)")
conn.execute("INSERT INTO Employees VALUES (3, 'Charlie', 'Engineering', 75000)")
conn.commit() # Save the data insertion

# 2. Write the SQL Query
sql_query = "SELECT Name, Salary FROM Employees WHERE Department = 'Engineering';"

# 3. Use Pandas to run the query and load directly into a DataFrame
df_salaries = pd.read_sql_query(sql_query, conn)

print(df_salaries)

# 4. Close the connection when finished
conn.close()

#Change the sql_query variable to select all columns (*) from the Employees table.
import pandas as pd
import sqlite3
conn = sqlite3.connect(':memory:')
create_table_query = """
CREATE TABLE Employees (
    ID INT,
    Name TEXT,
    Department TEXT,
    Salary INT
);
"""
conn.execute(create_table_query)
conn.execute("INSERT INTO Employees VALUES (1, 'Alice', 'Engineering', 70000)")
conn.execute("INSERT INTO Employees VALUES (2, 'Bob', 'Sales', 60000)")
conn.execute("INSERT INTO Employees VALUES (3, 'Charlie', 'Engineering', 75000)")
conn.commit()
sql_query = "SELECT * FROM Employees WHERE Salary > 65000 ORDER BY Salary DESC;"
df_filtered_employees = pd.read_sql_query(sql_query, conn)
print(df_filtered_employees)

#Lesson 15: Introduction to Version Control with Git and GitHub
#Lesson 16: Basic Machine Learning (Scikit-learn)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Create a sample dataset of study hours vs final exam scores
data = {'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Exam_Score': [40, 45, 55, 60, 70, 75, 85, 88, 92, 95]}
df = pd.DataFrame(data)

# Define X and Y (X needs to be 2D for scikit-learn)
X = df[['Hours_Studied']]
y = df['Exam_Score']

# Split data into 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model and train it
model = LinearRegression()
model.fit(X_train, y_train) # This is the learning step!

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model (e.g., Mean Absolute Error)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, predictions))
print("R-squared score:", metrics.r2_score(y_test, predictions))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

student_data = {
    'Math_Score':,
    'Science_Score':
}
students_df = pd.DataFrame(student_data)

X = students_df[['Math_Score']]
y = students_df['Science_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("R-squared score:", metrics.r2_score(y_test, predictions))

import kagglehub
path = kagglehub.dataset_download("neurocipher/student-performance")
print("Path to dataset files:", path)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("C/kaggle/input/student-performance/StudentPerformance.csv")
df = pd.DataFrame("/kaggle/input/student-performance/StudentPerformance.csv")