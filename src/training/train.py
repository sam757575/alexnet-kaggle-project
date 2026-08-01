from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data.dataset import Dataset
from src.models.alexnet import AlexNet
from src.utils.helpers import plot_training_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_MODEL_PATH = PROJECT_ROOT / 'best_model.h5'
TRAINING_HISTORY_PATH = PROJECT_ROOT / 'training_history.png'


def train_model(epochs=1000, batch_size=32):
    # Load and preprocess the dataset
    dataset = Dataset()
    train_data, val_data = dataset.load_data(batch_size=batch_size)

    # Initialize the AlexNet model
    model = AlexNet()
    model.compile_model()

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(str(BEST_MODEL_PATH), save_best_only=True)

    # Train the model
    # Note: batch_size is not passed here because train_data/val_data are
    # tf.data.Dataset objects that are already batched (see Dataset.load_data).
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
    )

    plot_training_history(history, save_path=str(TRAINING_HISTORY_PATH), show=False)
    print(f"Training history plot saved to {TRAINING_HISTORY_PATH}")

    return history


def evaluate_model():
    # Load the best model
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    # Load and preprocess the validation dataset
    dataset = Dataset()
    _, val_data = dataset.load_data()

    # Evaluate the model
    results = model.evaluate(val_data)
    print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")


if __name__ == "__main__":
    train_model()
    evaluate_model()
