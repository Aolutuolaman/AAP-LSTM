"""Build the standard LSTM architecture used by the supplied checkpoints.

Architecture and vocabulary compatibility do not confer rights to a checkpoint.
See README.md for scientific provenance and license scope.
"""
from pathlib import Path


def load_predictor(checkpoint):
    """Load legacy Keras HDF5 weights without executing serialized model code."""
    import tensorflow as tf

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None, 23)),
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3, name="lstm"),
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5, name="lstm_1"),
        tf.keras.layers.Dense(23, activation="softmax", name="dense"),
    ], name="aap_lstm_inference")
    model.load_weights(str(checkpoint))
    return model
