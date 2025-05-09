import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from data_preprocessing import AudioPreprocessor

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(x))
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices].copy()
        batch_y = self.y[batch_indices].copy()
        
        # Apply random augmentations
        for i in range(len(batch_x)):
            # Random amplitude scaling
            scale = np.random.uniform(0.9, 1.1)
            batch_x[i] = batch_x[i] * scale
            
            # Random noise addition (reduced intensity)
            noise = np.random.normal(0, 0.002, batch_x[i].shape)
            batch_x[i] = batch_x[i] + noise
            
            # Random time shift (reduced range)
            shift = np.random.randint(-5, 5)
            if shift > 0:
                batch_x[i] = np.pad(batch_x[i], ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
            elif shift < 0:
                batch_x[i] = np.pad(batch_x[i], ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def create_model(input_shape, training=True):
    """Create a simplified model architecture focusing on essential features."""
    model = tf.keras.Sequential([
        # Input and normalization
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LayerNormalization(),
        
        # First conv block
        tf.keras.layers.Conv1D(32, 7, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Second conv block
        tf.keras.layers.Conv1D(64, 5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Third conv block
        tf.keras.layers.Conv1D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Dense layers
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

def train_model():
    """Train the improved baby cry classifier model"""
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=44100, duration=5.0)
    
    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    X, y = preprocessor.load_dataset("data")
    
    # Ensure X is 3D (samples, timesteps, features)
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
    
    # Normalize the input data
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / (std + 1e-8)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print("\nClass weights:", class_weight_dict)
    print(f"Input shape: {X_train.shape}")
    
    # Create training model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), training=True)
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Create data generator for training
    train_generator = AudioDataGenerator(X_train, y_train, batch_size=32)
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save final model
    model.save('models/baby_cry_model.h5')
    
    # Evaluate final model
    print("\nEvaluating final model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_accuracy:.2%}")
    
    return model, history

if __name__ == "__main__":
    train_model() 