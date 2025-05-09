# Baby Cry Analyzer

A web application that analyzes baby cries to help parents understand their baby's needs. The application uses deep learning to classify baby cries into six categories: belly pain, burping, discomfort, hungry, tired, and processed.

## Features

- Upload audio files (.wav) of baby cries
- Real-time analysis using deep learning
- Confidence scores for each category
- Easy-to-use web interface
- Instant results

## Live Demo

The application is deployed and accessible at: https://baby-cry-analyzer.streamlit.app

## Technologies Used

- Python
- PyTorch
- Streamlit
- Librosa (for audio processing)
- Soundfile
- NumPy
- Scikit-learn

## Local Development

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kulzum624/baby-cry-analyzer.git
cd baby-cry-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application locally:
```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. Visit the application URL
2. Click "Choose an audio file..." button
3. Select a .wav file containing a baby's cry
4. Wait for the analysis results
5. View the predicted cry type and confidence scores

## Model Information

The application uses a deep learning model trained on various baby cry samples. The model can classify cries into six different categories with associated confidence scores.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 