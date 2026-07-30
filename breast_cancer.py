from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
cancer = load_breast_cancer()

# Convert to DataFrame
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Add target column
data['target'] = cancer.target

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Predict first patient
sample = X.iloc[[0]]
prediction = model.predict(sample)

if prediction[0] == 0:
    print("\nPrediction: Malignant")
else:
    print("\nPrediction: Benign")