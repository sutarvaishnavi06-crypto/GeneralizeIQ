import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

SEED = 42
VALIDATION_SIZE = 0.10

print("Loading Fashion-MNIST...")
(X_original, y_original), (X_test, y_test) = fashion_mnist.load_data()

X_train, X_val, y_train, y_val = train_test_split(
    X_original,
    y_original,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=y_original
)

print("Normalizing pixel values...")
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nChecking data integrity...")
assert X_train.shape[0] == 54000
assert X_val.shape[0] == 6000
assert X_test.shape[0] == 10000
assert X_train.min() >= 0 and X_train.max() <= 1
assert X_val.min() >= 0 and X_val.max() <= 1
assert X_test.min() >= 0 and X_test.max() <= 1

np.savez_compressed(
    "data/processed_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\n" + "=" * 45)
print("GENERALIZEIQ - DATA PREPARATION")
print("=" * 45)

print(f"Training set   : {X_train.shape[0]} images")
print(f"Validation set : {X_val.shape[0]} images")
print(f"Test set      : {X_test.shape[0]} images")

print("\nImage shape:", X_train.shape[1:])
print("Normalized range:", X_train.min(), "to", X_train.max())
print("Data type:", X_train.dtype)

print("\nData saved to data/processed_data.npz")
print("Preparation completed successfully!")