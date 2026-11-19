
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"

data = np.load(ROOT / "data" / "processed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

model = load_model(RESULTS / "baseline_model.keras")

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("\n" + "=" * 50)
print("GENERALIZEIQ - BASELINE TEST EVALUATION")
print("=" * 50)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

report = classification_report(
    y_test,
    predictions,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print(report)

with open(RESULTS / "baseline_test_report.txt", "w") as file:
    file.write(f"Test Loss: {loss:.4f}\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

matrix = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(matrix, cmap="Blues")

ax.set_title("Baseline - Test Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)

for i in range(10):
    for j in range(10):
        ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)

fig.colorbar(image)
plt.tight_layout()
plt.savefig(RESULTS / "baseline_confusion_matrix.png", dpi=300)
plt.show()

print("\nBaseline evaluation saved successfully!")

