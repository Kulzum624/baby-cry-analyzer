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
tab1, tab2 = st.tabs(["Upload Audio", "Instructions"])

with tab1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.write("Upload Baby's Cry Audio")
    uploaded_file = st.file_uploader("", type=['wav', 'mp3'], key="file_uploader")
    st.write("Supported formats: WAV, MP3 • Max size: 200MB")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Analyze Audio"):
            st.session_state.analyze_file = True
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.write("How to Use:")
    st.markdown("""
    1. Record your baby's cry using any recording app on your phone or computer
    2. Save the recording as a WAV or MP3 file
    3. Upload the file using the 'Upload Audio' tab
    4. Click 'Analyze Audio' to get results
    
    Tips for best results:
    - Record in a quiet environment
    - Keep the microphone close to your baby
    - Record for at least 3-5 seconds
    - Avoid background noise
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Process audio file
if uploaded_file is not None and st.session_state.get('analyze_file', False):
    st.markdown('<div class="results-box">', unsafe_allow_html=True)
    with st.spinner("Analyzing cry pattern..."):
        try:
            # Create a temporary file if using uploaded file
            if isinstance(uploaded_file, str):
                audio_path = uploaded_file
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
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
            if not isinstance(uploaded_file, str):
                os.unlink(audio_path)
                
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