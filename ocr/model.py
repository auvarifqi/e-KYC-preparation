from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block

def build_ocr_model(input_shape, output_dim, activation="leaky_relu", dropout=0.2):
    """
    Build a convolutional OCR model using residual blocks and bidirectional LSTM layers.

    Args:
        input_shape (tuple): Shape of the input images.
        output_dim (int): Number of output classes.
        activation (str): Activation function for the convolutional layers.
        dropout (float): Dropout rate for regularization.

    Returns:
        keras.models.Model: Compiled OCR model.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name="input")

    # Normalize input images
    normalized_inputs = layers.Lambda(lambda x: x / 255)(inputs)

    # First residual block
    x1 = residual_block(normalized_inputs, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    # Second residual block
    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Third residual block
    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Fourth residual block
    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Reshape for bidirectional LSTM
    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    # Bidirectional LSTM layer
    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(squeezed)

    # Output layer
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=output)
    return model
