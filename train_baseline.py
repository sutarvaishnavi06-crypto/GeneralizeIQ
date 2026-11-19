
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models.baseline import build_baseline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_data.npz"
RESULTS = ROOT / "reports" / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

model = build_baseline()

print("=" * 50)
print("GENERALIZEIQ - BASELINE TRAINING")
print("=" * 50)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.index += 1
history_df.index.name = "epoch"
history_df.to_csv(RESULTS / "baseline_history.csv")

model.save(RESULTS / "baseline_model.keras")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Baseline Learning Curves - Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Baseline Learning Curves - Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "baseline_learning_curves.png", dpi=300)
plt.show()

print("\nBaseline training completed!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Lowest validation loss: {min(history.history['val_loss']):.4f}")
print(f"Results saved in: {RESULTS}")
