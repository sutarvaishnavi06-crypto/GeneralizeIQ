
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.regularizers import l2


def build_l2_model(weight=0.0001):
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),

        Dense(256, activation="relu", kernel_regularizer=l2(weight)),
        Dense(128, activation="relu", kernel_regularizer=l2(weight)),
        Dense(64, activation="relu", kernel_regularizer=l2(weight)),

        Dense(10, activation="softmax")
    ], name="L2_Network")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
