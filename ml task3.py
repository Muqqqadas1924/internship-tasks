import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:/Users/LENOVO/Desktop/AI datasets/Customer Churn.csv')

print(df.columns)
print(df.head())
print(df.shape)

X = df[['Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
       'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
       'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age',
       'Customer Value']]
y = df['Churn']

print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title("Count of Churned vs Non-Churned Customers")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 6))
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'orange'])
plt.title("Churn vs Non-Churn (Pie Chart)")
plt.ylabel("")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Subscription  Length', y='Charge  Amount', hue='Churn', data=df)
plt.title("Subscription Length vs Charge Amount by Churn")
plt.xlabel("Subscription Length")
plt.ylabel("Charge Amount")
plt.show()
