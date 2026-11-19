
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"

EXPERIMENTS = {
    "Baseline": ("baseline_history.csv", "baseline_test_report.txt"),
    "Dropout": ("dropout_history.csv", "dropout_test_report.txt"),
    "L2": ("l2_history.csv", "l2_test_report.txt"),
    "Early Stopping": (
        "early_stopping_history.csv",
        "early_stopping_test_report.txt"
    )
}

def extract_number(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None

records = []

for name, (history_file, report_file) in EXPERIMENTS.items():
    history_path = RESULTS / history_file
    report_path = RESULTS / report_file

    if not history_path.exists() or not report_path.exists():
        print(f"Skipping {name}: required files missing")
        continue

    history = pd.read_csv(history_path)
    report = report_path.read_text()

    test_accuracy = extract_number(
        report, r"Test Accuracy\s*:\s*([\d.]+)"
    )
    test_loss = extract_number(
        report, r"Test Loss\s*:\s*([\d.]+)"
    )

    records.append({
        "Experiment": name,
        "Epochs": len(history),
        "Train Accuracy": history["accuracy"].iloc[-1],
        "Validation Accuracy": history["val_accuracy"].iloc[-1],
        "Best Validation Loss": history["val_loss"].min(),
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
        "Generalization Gap": (
            history["accuracy"].iloc[-1] - history["val_accuracy"].max()
        )
    })

if not records:
    raise FileNotFoundError("No complete experiment files were found.")

comparison = pd.DataFrame(records)
comparison = comparison.sort_values("Test Accuracy", ascending=False)

comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

print("\nGENERALIZEIQ - MODEL COMPARISON")
print("=" * 75)
print(comparison.round(4).to_string(index=False))

best = comparison.iloc[0]
print("\nBEST MODEL:", best["Experiment"])
print("TEST ACCURACY:", round(best["Test Accuracy"], 4))
print("TEST LOSS:", round(best["Test Loss"], 4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(comparison["Experiment"], comparison["Test Accuracy"])
axes[0].set_title("Test Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(comparison["Experiment"], comparison["Generalization Gap"])
axes[1].set_title("Generalization Gap Comparison")
axes[1].set_ylabel("Accuracy Gap")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "experiment_comparison.png", dpi=300)
plt.show()

print("\nComparison engine completed!")
