import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\patel\OneDrive\Desktop\DataAnalytics\Task3\Wine_Quality\WineQT.csv')

print("First few rows of the dataset:")
print(data.head())
data.dropna(inplace=True)

X = data.drop('quality', axis=1) 
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)

svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("SGD Accuracy:", accuracy_score(y_test, y_pred_sgd))
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
print("\nClassification Report for SGD:\n", classification_report(y_test, y_pred_sgd, zero_division=0))
print("\nClassification Report for SVC:\n", classification_report(y_test, y_pred_svc, zero_division=0))

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Wine Quality Dataset')
plt.show()
