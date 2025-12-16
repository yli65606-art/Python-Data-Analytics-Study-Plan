#Student performance project

#download and load the datasets
import pandas as pd
import os

from sklearn import metrics
from sklearn.linear_model import LinearRegression
import kagglehub
path = kagglehub.dataset_download("neurocipher/student-performance")
print("Path to dataset files:", path)

csv_file_path = os.path.join(path, 'StudentPerformance.csv')
df = pd.read_csv(csv_file_path)

#Extracting info of the loaded dataset
df.head()
df.info()
df.describe()

#Data Cleaning: Checking for NULL values in the DataFrame
df.isnull().sum()#No NULL value in the Dataframe

#Rename columens to standard format
df.rename(columns={
    'Hours Studied': 'hours_studied',
    'Previous Scores': 'previous_scores',
    'Extracurricular Activities': 'extracurricular_activities',
    'Sleep Hours': 'sleep_hours',
    'Sample Question Papers Practiced': 'sample_question_papers_practiced',
    'Performance Index': 'performance_index'}, inplace=True)

#Transform object DP to int DP
df['extracurricular_activities'] = (df['extracurricular_activities'] == 'Yes').astype(int)

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#scatterplot PI vs PS
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='performance_index', y='previous_scores')
plt.title("performance index vs previous scores")
plt.show()
#scatterplot PI vs HS
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='performance_index', y='hours_studied')
plt.title("performance index vs hours_studied")
plt.show()
#scatterplot PI vs HS
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='performance_index', y='extracurricular_activities')
plt.title("performance index vs extracurricular_activities")
plt.show()
#scatterplot PI vs SH
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='performance_index', y='sleep_hours')
plt.title("performance index vs sleep_hours")
plt.show()
#scatterplot PI vs SQPP
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='performance_index', y='sample_question_papers_practiced')
plt.title("performance index vs sample_question_papers_practiced")
plt.show()

#simplify the code with for loop
custom_colors = ['red', 'black']
feature_columns = [
    'previous_scores',
    'hours_studied',
    'extracurricular_activities',
    'sleep_hours',
    'sample_question_papers_practiced'
]
for column_name in feature_columns:
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='performance_index', y=column_name, hue='extracurricular_activities', palette=custom_colors)
    plt.title(f"Performance Index vs. {column_name.replace('_', ' ').title()}")
    plt.show()

#kaggle code boxplot
fig,axes= plt.subplots(3,2,figsize=(12,10))
axes=axes.flatten()
fig.delaxes(axes[-1])

colour=['purple','coral','red','green','blue','cyan']
columns=list(df.columns)
del columns[-1]

for feature in range(len(columns)):
    sns.boxplot(x=df[columns[feature]], ax=axes[feature], color=colour[feature])
    axes[feature].set_title(f"Attribute : {columns[feature]}",fontweight='bold')
    axes[feature].set_xlabel(f"{columns[feature]}",fontweight='bold')

plt.suptitle("Visualization before Outlier Treatment\n", fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

#Separate the data into inputs(x) and the target(y)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
features = ['hours_studied', 'previous_scores', 'extracurricular_activities', 'sleep_hours', 'sample_question_papers_practiced']
x = df[features]
y = df['performance_index']

#Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#initialize the linearRegression model:
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")


