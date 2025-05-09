import streamlit as st
import torch
from model import BabyCryClassifier
from src.data_preprocessing import AudioPreprocessor
import tempfile
import os
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Baby Cry Analyzer",
    page_icon="👶",
    layout="centered"
)

# Custom CSS to match App.tsx style
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: white;
    }
    .main {
        max-width: 800px !important;
        padding: 2rem;
        margin: 0 auto;
    }
    .stTitle {
        color: white;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-box {
        background-color: #2d2d2d;
        border: 2px dashed #666;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .results-box {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = BabyCryClassifier(num_classes=6)
    model.load_state_dict(torch.load('final_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Initialize preprocessor
preprocessor = AudioPreprocessor(sample_rate=44100, duration=5.0)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

st.title("Baby Cry Analyzer")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

with tab1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.write("Drag and drop file here")
    uploaded_file = st.file_uploader("", type=['wav'], key="file_uploader")
    st.write("Limit 200MB per file • WAV")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.write("Record Baby's Cry")
    
    # Using Streamlit's built-in audio recorder
    audio_bytes = st.audio_recorder()
    
    if audio_bytes is not None:
        # Save the recorded audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            st.session_state.recorded_file = tmp_file.name
            st.success("Recording saved! Click 'Analyze' to process.")
            
        if st.button("Analyze Recording"):
            st.session_state.analyze_recording = True
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process audio (either uploaded or recorded)
audio_file = uploaded_file if uploaded_file is not None else (
    st.session_state.get('recorded_file') if 'recorded_file' in st.session_state 
    and getattr(st.session_state, 'analyze_recording', False) else None
)

if audio_file is not None:
    st.markdown('<div class="results-box">', unsafe_allow_html=True)
    with st.spinner("Analyzing cry pattern..."):
        try:
            # Create a temporary file if using uploaded file
            if isinstance(audio_file, str):
                audio_path = audio_file
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    audio_path = tmp_file.name

            # Extract features
            audio = preprocessor.extract_features(audio_path)
            audio = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(audio)
                probabilities = torch.softmax(prediction, dim=1)
            
            # Get results
            class_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'processed']
            predicted_class = torch.argmax(probabilities[0]).item()
            predicted_label = class_labels[predicted_class].replace('_', ' ').title()
            
            # Display results
            st.markdown(f"### Predicted Cry Type: {predicted_label}")
            
            # Show confidence scores
            st.write("Confidence Scores:")
            for label, prob in zip(class_labels, probabilities[0]):
                prob_value = float(prob)
                st.write(f"{label.replace('_', ' ').title()}: {prob_value:.1%}")
                st.progress(prob_value)
            
            # Cleanup
            if not isinstance(audio_file, str):
                os.unlink(audio_path)
            elif 'recorded_file' in st.session_state:
                os.remove(st.session_state.recorded_file)
                del st.session_state.recorded_file
                st.session_state.analyze_recording = False
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Baby Cry Analyzer • AI-Powered Analysis</p>
    </div>
""", unsafe_allow_html=True) 