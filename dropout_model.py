
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout


def build_dropout_model(rate=0.30):
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),

        Dense(256, activation="relu"),
        Dropout(rate),

        Dense(128, activation="relu"),
        Dropout(rate),

        Dense(64, activation="relu"),
        Dropout(rate),

        Dense(10, activation="softmax")
    ], name="Dropout_Network")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

