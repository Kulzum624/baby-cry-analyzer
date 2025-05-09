import librosa
import numpy as np
import os
import soundfile as sf
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 44100, duration: float = 5.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.label_encoder = LabelEncoder()

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from an audio file."""
        try:
            # Load audio file
            try:
                audio, file_sr = sf.read(audio_path)
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                # Resample if necessary
                if file_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sample_rate)
            except Exception as e:
                print(f"SoundFile failed, trying librosa: {str(e)}")
                # Fallback to librosa
                audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                audio = np.pad(audio, (0, max(0, self.n_samples - len(audio))))
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Return raw audio waveform
            return audio
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def load_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the entire dataset."""
        features = []
        labels = []
        
        # Process each class directory
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Processing {label} files...")
            for audio_file in os.listdir(class_dir):
                if not audio_file.endswith(('.wav', '.mp3')):
                    continue
                    
                audio_path = os.path.join(class_dir, audio_file)
                feature = self.extract_features(audio_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(label)
        
        # Convert labels to numerical format
        y = self.label_encoder.fit_transform(labels)
        X = np.array(features)
        
        return X, y

    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the dataset using SMOTE."""
        # Reshape the data for SMOTE
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(n_samples, -1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
        
        # Reshape back to original format
        X_balanced = X_balanced.reshape(-1, n_timesteps, n_features)
        
        return X_balanced, y_balanced

    def augment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation techniques to audio."""
        augmented = []
        
        # Time stretching
        stretch_rates = [0.8, 1.2]
        for rate in stretch_rates:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            if len(stretched) > self.n_samples:
                stretched = stretched[:self.n_samples]
            else:
                stretched = np.pad(stretched, (0, max(0, self.n_samples - len(stretched))))
            augmented.append(stretched)
        
        # Pitch shifting
        pitch_shifts = [-2, 2]
        for steps in pitch_shifts:
            shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
            augmented.append(shifted)
        
        # Add noise
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, audio.shape)
        noisy_audio = audio + noise
        augmented.append(noisy_audio)
        
        return augmented 