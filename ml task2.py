import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('C:/Users/LENOVO/Desktop/AI datasets/messages.csv')

df['subject'] = df['subject'].fillna('no subject') 
df.isnull().sum()
df['combined_text'] = df['subject'] + " " + df['message']

X = df['combined_text']
y = df['label']

df = df.drop_duplicates()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# SVM model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("Naive Bayes Accuracy: ", nb_accuracy)
print("SVM Accuracy: ", svm_accuracy)

print("Naive Bayes Classification Report: ")
print(classification_report(y_test, y_pred_nb))

print("SVM Classification Report: ")
print(classification_report(y_test, y_pred_svm))

conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6,4))
models = ['Naive Bayes', 'SVM']
accuracies = [nb_accuracy, svm_accuracy]
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Naive Bayes')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: SVM')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
df_report_nb = pd.DataFrame(report_nb).transpose()
df_report_nb.iloc[:-1, :-1].plot(kind='bar', figsize=(10,6))
plt.title('Naive Bayes Classification Report')
plt.show()

report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
df_report_svm = pd.DataFrame(report_svm).transpose()
df_report_svm.iloc[:-1, :-1].plot(kind='bar', figsize=(10,6))
plt.title('SVM Classification Report')
plt.show()
