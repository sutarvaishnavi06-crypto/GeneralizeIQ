import numpy as np
from tensorflow.keras.datasets import fashion_mnist

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("=" * 45)
print("       GENERALIZEIQ - DATASET INSPECTION")
print("=" * 45)

print("\nDataset Shapes:")
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

print("\nImage Information:")
print("Image dimensions:", X_train.shape[1:])
print("Pixel range:", X_train.min(), "to", X_train.max())
print("Data type:", X_train.dtype)
print("Number of classes:", len(class_names))

print("\nClass Distribution in Training Data:")
for class_id, class_name in enumerate(class_names):
    count = np.sum(y_train == class_id)
    print(f"{class_name:<15}: {count}")

print("\nDataset loaded and inspected successfully!")