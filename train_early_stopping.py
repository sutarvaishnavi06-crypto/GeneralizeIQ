
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from models.baseline import build_baseline

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

data = np.load(ROOT / "data" / "processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

model = build_baseline()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

print("=" * 50)
print("GENERALIZEIQ - EARLY STOPPING EXPERIMENT")
print("=" * 50)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

pd.DataFrame(history.history).to_csv(
    RESULTS / "early_stopping_history.csv", index=False
)
model.save(RESULTS / "early_stopping_model.keras")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("Early Stopping - Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Early Stopping - Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "early_stopping_learning_curves.png", dpi=300)
plt.show()

best_epoch = np.argmin(history.history["val_loss"]) + 1

print("\nEarly Stopping experiment completed!")
print(f"Epochs executed: {len(history.history['loss'])}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
