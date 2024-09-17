import os
import numpy as np
from typing import Optional
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model as keras_load_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from io import BytesIO
import google.generativeai as genai

load_dotenv()

# Configure Gemini AI
api_key = os.getenv("API_KEY")  # Fetch API key from environment variable
if api_key is None:
    raise ValueError("API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# Load model once at the start
lung_disease_model = keras_load_model('lung_disease_model.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


async def process_message_and_image(message: str, file: Optional[UploadFile]) -> dict:
    if file:
        # Handle image file
        print(0)
        try:
            if file.content_type not in ["image/jpeg", "image/png"]:
                return {"error": "Unsupported file format. Please upload a JPEG or PNG image."}

            file_content = await file.read()
            img = Image.open(BytesIO(file_content))  # Wrap the byte content in BytesIO
            img_array = preprocess_image(img)
            print(1)
            # Predict using the pre-loaded model
            predictions = lung_disease_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            labels = ['Covid-19', 'Normal', 'Viral Pneumonia', 'Bacterial Pneumonia']
            predicted_label = labels[predicted_class]
            
            return {"message": f"Prediction for lung disease: {predicted_label}"}
        except Exception as e:
            print(str(e))
            return {"error": f"Error processing image: {str(e)}"}
    
    if message:
        # Handle text message
        try:
            prompt = (
                "Forget the previous chat. You are an AI doctor assistant. A new patient is describing their symptoms, "
                "and you are helping them to understand what possible disease they might have, "
                "so they can consult a doctor. Respond carefully based on the symptoms provided."
                "Always keep your response short and to the point."
            )
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{prompt}\n\nSymptoms: {message}")
            return {"message": response.text}
        except Exception as e:
            return {"error": f"Error processing message: {str(e)}"}
    
    return {"message": "No image provided and no symptoms described. Please describe your symptoms or upload an image for better diagnosis."}

@app.post("/predict")
async def predict(message: str = Form(...), file: UploadFile = File(None)):
    result = await process_message_and_image(message, file)
    return result
