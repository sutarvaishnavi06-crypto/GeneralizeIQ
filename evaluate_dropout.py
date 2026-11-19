
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"

data = np.load(ROOT / "data" / "processed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

model = load_model(RESULTS / "dropout_model.keras")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

report = classification_report(
    y_test, predictions, target_names=classes, digits=4
)

print("\n" + "=" * 45)
print("GENERALIZEIQ - DROPOUT TEST EVALUATION")
print("=" * 45)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("\nClassification Report:")
print(report)

with open(RESULTS / "dropout_test_report.txt", "w") as file:
    file.write(f"Test Loss: {loss:.4f}\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

print("\nDropout evaluation saved successfully!")
