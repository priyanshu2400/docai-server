import os
import numpy as np
import pandas as pd
from typing import Optional
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model as keras_load_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from io import BytesIO
import joblib
import google.generativeai as genai
import json
load_dotenv()

# Configure Gemini AI
api_key = os.getenv("API_KEY")  # Fetch API key from environment variable
if api_key is None:
    raise ValueError("API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# Load models
lung_disease_model = keras_load_model("/code/app/lung_disease_model.h5")

# lung_disease_model = keras_load_model('lung_disease_model.h5')
heart_disease_model = joblib.load('/code/app/disease_prediction_model.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def preprocess_image(img: Image.Image) -> np.ndarray:
    try:
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((150, 150))  # Resize to match the model's expected input
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")


def predict_heart_disease(data: dict) -> str:
    # Prepare data for prediction
    df = pd.DataFrame([data])
    
    # Predict using the heart disease model
    prediction = heart_disease_model.predict(df)
    disease_status = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return disease_status


async def process_message_and_image(message: str, file: Optional[UploadFile]) -> dict:
    if file:
        # Handle lung disease prediction
        try:
            if file.content_type not in ["image/jpeg", "image/png"]:
                return {"error": "Unsupported file format. Please upload a JPEG or PNG image."}

            file_content = await file.read()
            img = Image.open(BytesIO(file_content))  # Wrap the byte content in BytesIO
            img_array = preprocess_image(img)
            
            # Predict using the pre-loaded lung model
            predictions = lung_disease_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            labels = ['Covid-19', 'Normal', 'Viral Pneumonia', 'Bacterial Pneumonia']
            predicted_label = labels[predicted_class]
            
            response = f"Prediction for lung disease: {predicted_label}. Please consult a doctor for better understanding."
            
            # Use Gemini to format the response
            prompt = f"Please convert the following prediction into a human-friendly format: {response}. Also, recommend consulting a doctor."
            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_response = model.generate_content(prompt)
            return {"message": gemini_response.text}
        
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

    elif message:
        # Handle text message
        try:
            prompt = (
                "Forget the previous chat. You are an AI doctor assistant. A new patient is describing their symptoms, "
                "and you are helping them to understand what possible disease they might have, "
                "so they can consult a doctor. Respond carefully based on the symptoms provided."
                "Always keep your response short and to the point."
            )
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f" {message} \n \n does the above have all the folllowing fields? ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] reply in yes or no")
            # Check if all required fields are present
            if response.text.strip().lower() == "yes":
                data = {
                    'age': [63],
                    'sex': [1],
                    'cp': [3],
                    'trestbps': [145],
                    'chol': [233],
                    'fbs': [1],
                    'restecg': [0],
                    'thalach': [150],
                    'exang': [0],
                    'oldpeak': [2.3],
                    'slope': [0],
                    'ca': [0],
                    'thal': [1]
                }
                print(2)
                # Ask Gemini to extract data from the message and format it as a DataFrame
                response = model.generate_content(f"{message} \n \n Extract the data exactly in the form of: {data} and give as output")
                l = len(response.text)
                substring = response.text[7:l-5]
                print(2.1)
                try:
                    json_string = substring.replace("'", '"')
                    data = json.loads(json_string)
                except Exception as e:
                    print(str(e))
                print(2.2)
                df = pd.DataFrame(data)
                print(1)
                if isinstance(df, pd.DataFrame):
                    print(3)
                    try:
                        # Load the trained model
                        model = joblib.load('disease_prediction_model.pkl')
                        
                        # Make the prediction
                        prediction = model.predict(df)

                        # Output the prediction result
                        if prediction[0] == 1:
                            response = "The model predicts that the patient has heart disease."
                        else:
                            response = "The model predicts that the patient does not have heart disease."
                    
                    except Exception as e:
                        print(str(e))
                        response = f"An error occurred: {str(e)}"
                else:
                    response = "Gemini was unable to extract the data or the input message is not in the correct format."

                return {"message": response}
            else:
                # Ask for further input
                # Generate body part response based on symptoms
                body_part_response = model.generate_content(f"{prompt} \n \n symptoms:{message} \n \n Based on symptoms, which part of the body is most likely affected: lungs or heart or something else. Give one word answer like heart, lungs, brain, or neither.")

                # Safely extract the text from the response
                response_text = body_part_response.text.lower()
                print(response_text)
                # Check the response for heart, lungs, or neither
                if 'heart' in response_text:
                    return {"message": "It seems like the issue might be with the heart. Please provide the following details: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal."}
                elif 'lungs' in response_text:
                    return {"message": "It seems like the issue might be with the lungs. Please upload scans for better diagnosis."}
                elif 'neither' in response_text:
                    # Generate a follow-up response if neither heart nor lungs are indicated
                    detailed_response = model.generate_content(f"You are an AI doctor assistant. A new patient is describing their symptoms, "
                                                                "and you are helping them to understand what possible disease they might have, "
                                                                "so they can consult a doctor. Respond carefully based on the symptoms provided."
                                                            "very important :- keep your response short and to the point . \n {message}")
                    return {"message":detailed_response.text}
                else:
                    return {"message": f"Based on the symptoms, the problem might be related to {response_text}. However, I cannot provide a detailed analysis. Please consult a doctor."}
        except Exception as e:
            print(str(e))
            return {"error": f"Error processing message: {str(e)}"}

    return {"message": "No image provided and no symptoms described. Please describe your symptoms or upload an image for better diagnosis."}

@app.get("/")
def read_root():
    return {"Hello": "World"}
     
@app.post("/predict")
async def predict(message: str = Form(...), file: UploadFile = File(None)):
    result = await process_message_and_image(message, file)
    return result
