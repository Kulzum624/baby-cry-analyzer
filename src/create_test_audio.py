import numpy as np
import soundfile as sf
import os

def create_test_audio():
    """Create a test audio file with a simple sine wave."""
    # Create output directory if it doesn't exist
    os.makedirs('data/belly_pain', exist_ok=True)
    
    # Generate a 5-second sine wave at 440 Hz (A4 note)
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Save the audio file
    output_path = 'data/belly_pain/belly_pain_001.wav'
    sf.write(output_path, audio, sample_rate)
    print(f"Created test audio file at: {output_path}")

if __name__ == "__main__":
    create_test_audio() 