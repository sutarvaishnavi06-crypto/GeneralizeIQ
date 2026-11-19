from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense


def build_baseline():
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),

        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),

        Dense(10, activation="softmax")
    ], name="Baseline_Network")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model