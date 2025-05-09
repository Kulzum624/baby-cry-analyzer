import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class BabyCryModel:
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self) -> models.Model:
        """Build and return the model architecture."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Conv block
            layers.Conv1D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Second Conv block
            layers.Conv1D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Third Conv block
            layers.Conv1D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # LSTM layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train: tf.Tensor, y_train: tf.Tensor,
              X_val: tf.Tensor, y_val: tf.Tensor,
              batch_size: int = 32, epochs: int = 50) -> tf.keras.callbacks.History:
        """Train the model with early stopping."""
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """Make predictions on new data."""
        return self.model.predict(X)

    def save_model(self, path: str):
        """Save the model to disk."""
        self.model.save(path)

    @classmethod
    def load_model(cls, path: str) -> 'BabyCryModel':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(path)
        instance = cls(model.input_shape[1:], model.output_shape[1])
        instance.model = model
        return instance 