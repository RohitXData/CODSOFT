# titanic_survival.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
print("Dataset loaded. Total rows:", len(data))

# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age']]
print("Columns selected:", data.columns.tolist())

# Drop rows with missing values
data.dropna(inplace=True)
print("After dropping NA, rows left:", len(data))

# Encode 'Sex' as numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
print("Sex column encoded. Sample:\n", data.head())

# Features and target
X = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model training complete.")

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction complete. Model accuracy: {accuracy:.2f}")
