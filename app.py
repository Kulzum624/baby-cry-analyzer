import os
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.data_preprocessing import AudioPreprocessor
import uvicorn
from typing import Dict, Any
import json
from datetime import datetime
from model import BabyCryClassifier

app = FastAPI(title="Baby Cry Analyzer API")

# Add CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the audio preprocessor
preprocessor = AudioPreprocessor(sample_rate=44100, duration=5.0)

# Create a directory for temporary audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BabyCryClassifier(num_classes=6)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

def predict_cry(audio_path: str) -> Dict[str, Any]:
    """Predict the type of baby cry from an audio file."""
    try:
        print("Making prediction...")
        print(f"Preprocessing audio file: {audio_path}")
        
        # Extract raw audio waveform
        audio = preprocessor.extract_features(audio_path)
        print(f"Audio shape: {audio.shape}")
        
        # Convert to PyTorch tensor and add batch dimension
        audio = torch.from_numpy(audio).float()
        audio = audio.unsqueeze(0)  # Add batch dimension
        audio = audio.to(device)
        print(f"Audio tensor shape: {audio.shape}")
        
        # Make prediction
        print("Running model prediction...")
        with torch.no_grad():
            prediction = model(audio)
            probabilities = torch.softmax(prediction, dim=1)
        
        print(f"Raw prediction shape: {prediction.shape}")
        
        predicted_class = torch.argmax(probabilities[0]).item()
        print(f"Predicted class index: {predicted_class}")
        
        # Map class index to label (6 classes)
        class_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'processed', 'tired']
        predicted_label = class_labels[predicted_class]
        print(f"Predicted label: {predicted_label}")
        
        # Get confidence scores
        confidence_scores = {
            label: float(score) 
            for label, score in zip(class_labels, probabilities[0].cpu().numpy())
        }
        print(f"Confidence scores: {confidence_scores}")
        
        return {
            'predicted_class': predicted_label,
            'confidence_scores': confidence_scores,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in predict_cry: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Endpoint for audio file prediction."""
    try:
        print(f"\nReceived audio file: {file.filename}")
        
        # Create temp directory if it doesn't exist
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Save the uploaded file temporarily with proper extension
        temp_path = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        print(f"Saving to temporary file: {temp_path}")
        
        try:
            content = await file.read()
            with open(temp_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save audio file")
        
        print("Making prediction...")
        # Make prediction
        try:
            result = predict_cry(temp_path)
            print(f"Prediction result: {result}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process audio file")
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print("Temporary file removed")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "num_classes": 6
    }

if __name__ == "__main__":
    try:
        print("Starting Baby Cry Analyzer API server...")
        print(f"Using device: {device}")
        print("Model loaded successfully!")
        
        print("Starting server on http://localhost:8001")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise