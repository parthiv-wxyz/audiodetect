from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Input
)

def build_cnn(input_shape=(40, 138, 1)):

    model = Sequential([

        Input(shape=input_shape),

        Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(pool_size=(2,2)),


        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(pool_size=(2,2)),


        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(pool_size=(2,2)),


        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_cnn()

    model.summary()