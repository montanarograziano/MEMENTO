from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

DATA_DIR = Path('../../../data/models')


def get_merged_model():
    mri_model = get_model()
    pet_model = get_model()
    mri_model.load_weights(DATA_DIR / 'mri.h5')
    pet_model.load_weights(DATA_DIR / 'pet.h5')
    mri_model = Model(inputs=mri_model.input, outputs=mri_model.layers[-3].output)
    pet_model = Model(inputs=pet_model.input, outputs=pet_model.layers[-3].output)
    merged = concatenate([mri_model.output, pet_model.output])
    z = Dense(128, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    z = Dense(1, activation='sigmoid')(z)
    merged_model = Model(inputs=[mri_model.input, pet_model.input], outputs=z)
    return merged_model


def get_3d_model(width=128, height=128, depth=50):
    """Build a convolutional neural network model based on the Zunhair et al model.
    References
    ----------
    - https://arxiv.org/abs/2007.13224
    """
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3d-cnn")
    return model
