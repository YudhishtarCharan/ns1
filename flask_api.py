from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model/model.pkl')  # Adjust path if needed

# Map numeric prediction to species name
class_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return "🌼 Flask API for Iris Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features
        features = np.array(data['features']).reshape(1, -1)
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df = pd.DataFrame(features, columns=feature_names)

        # Make prediction
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]

        return jsonify({
            'prediction': prediction,
            'label': class_map[prediction],
            'probabilities': {class_map[i]: float(prob) for i, prob in enumerate(probabilities)}
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
