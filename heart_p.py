import pandas as pd
import joblib

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Prepare the data as a pandas DataFrame with correct feature names
data = pd.DataFrame({
    'age': [57],
    'sex': [0],
    'cp': [1],
    'trestbps': [130],
    'chol': [236],
    'fbs': [0],
    'restecg': [0],
    'thalach': [174],
    'exang': [0],
    'oldpeak': [0],
    'slope': [1],
    'ca': [1],
    'thal': [2]
})

# Make the prediction
prediction = model.predict(data)

# Output the prediction
print(f'Predicted Disease Status: {prediction[0]}')
