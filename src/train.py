import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import AudioPreprocessor
from model import BabyCryModel
import tensorflow as tf

def main():
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=22050, duration=3.0)
    
    # Load and preprocess dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    print("Loading dataset from:", data_dir)
    X, y = preprocessor.load_dataset(data_dir)
    
    # Balance dataset
    print("Balancing dataset...")
    X_balanced, y_balanced = preprocessor.balance_dataset(X, y)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Initialize and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y))
    model = BabyCryModel(input_shape, num_classes)
    
    print("Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=50
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'baby_cry_model.h5')
    print(f"Saving model to {model_path}")
    model.save_model(model_path)

if __name__ == "__main__":
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    main() 