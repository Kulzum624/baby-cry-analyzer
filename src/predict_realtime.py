import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os
from data_preprocessing import AudioPreprocessor
from model import BabyCryModel

class RealTimePredictor:
    def __init__(self, model_path: str, sample_rate: int = 22050, duration: float = 3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.preprocessor = AudioPreprocessor(sample_rate, duration)
        self.model = BabyCryModel.load_model(model_path)
        self.class_names = ['hungry', 'burping', 'belly_pain', 'discomfort', 'tired']

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio data to extract features."""
        # Ensure correct length
        if len(audio) > self.n_samples:
            audio = audio[:self.n_samples]
        else:
            audio = np.pad(audio, (0, max(0, self.n_samples - len(audio))))
        
        # Extract features (similar to AudioPreprocessor.extract_features)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        features = np.concatenate([
            mfccs,
            delta_mfccs,
            delta2_mfccs,
            spectral_centroid,
            spectral_rolloff
        ])
        
        return features.T

    def predict_cry(self, audio: np.ndarray) -> tuple:
        """Predict the type of cry from audio data."""
        # Process audio
        features = self.process_audio(audio)
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], confidence

    def callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print('Error:', status)
            return
        
        # Convert audio to mono and float32
        audio = np.mean(indata, axis=1).astype(np.float32)
        
        # Make prediction
        cry_type, confidence = self.predict_cry(audio)
        print(f"\rPredicted cry type: {cry_type} (confidence: {confidence:.2f})", end='')

    def start_listening(self):
        """Start listening to audio input."""
        try:
            with sd.InputStream(channels=1,
                              samplerate=self.sample_rate,
                              blocksize=self.n_samples,
                              callback=self.callback):
                print("Listening for baby cries... Press Ctrl+C to stop.")
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped listening.")
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    # Load model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'baby_cry_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize predictor
    predictor = RealTimePredictor(model_path)
    
    # Start listening
    predictor.start_listening()

if __name__ == "__main__":
    main() 