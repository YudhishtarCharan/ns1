import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read CSV file
data = pd.read_csv("data/iris.csv")
print(data.head())

# Check the distribution of species
print(data['species'].value_counts())

# Map species to numerical values
data['species'] = data['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

# Define features and labels (target variable)
X = data.drop('species', axis=1)
y = data['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
print(y_pred)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, 'model/model.pkl')

# Load the trained model (for testing purposes)
loaded_model = joblib.load('model/model.pkl')
