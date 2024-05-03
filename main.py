import pandas as pd

data = pd.read_csv("parkinsons/parkinsons.data", delimiter = ',')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = data.columns.drop(['name', 'status'])  # assuming 'status' is your label
data[features] = scaler.fit_transform(data[features])

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='status', y='MDVP:Fo(Hz)', data=data)
plt.title('Distribution of Fundamental Frequency')
plt.show()

from sklearn.model_selection import train_test_split

X = data.drop(['name', 'status'], axis=1)  # Drop non-feature columns
y = data['status']  # Assuming 'status' is the label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Detailed classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

from sklearn.svm import SVC

# Initialize SVM
svm_model = SVC(kernel='linear', C=1)  # Kernel can be 'linear', 'poly', 'rbf', 'sigmoid'

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("Classification Report:\n", classification_report(y_test, y_pred_svm))



