"""Modelo CNN baseline (mesma arquitetura dos notebooks de 2021), agora com
data augmentation opcional e dropout para reduzir o overfitting."""

import keras
from keras import layers

TAMANHO_IMG = (180, 180)


def camadas_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def cnn_baseline(n_classes: int, augmentation: bool = True, dropout: float = 0.3) -> keras.Model:
    """Retorna logits (sem softmax), igual ao modelo original."""
    entrada = keras.Input(shape=(*TAMANHO_IMG, 3))
    x = camadas_augmentation()(entrada) if augmentation else entrada
    x = layers.Rescaling(1.0 / 255)(x)
    for filtros in (16, 32, 64):
        x = layers.Conv2D(filtros, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    saida = layers.Dense(n_classes)(x)
    return keras.Model(entrada, saida, name="cnn_baseline")
