
import streamlit as st
import pandas as pd
from pathlib import Path
from components.insights import analyze_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "results"

st.set_page_config(
    page_title="GeneralizeIQ | Model Validation",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: #0b1020; color: #e6edf7; }
[data-testid="stMetric"] {
    background: #151d32;
    padding: 18px;
    border: 1px solid #273553;
    border-radius: 12px;
}
h1, h2, h3 { color: #e6edf7; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 GeneralizeIQ")
st.caption("Neural Network Validation & Regularization Intelligence")
st.divider()

comparison_file = RESULTS / "model_comparison.csv"

if not comparison_file.exists():
    st.error("Comparison results not found. Run the comparison engine first.")
    st.stop()

df = pd.read_csv(comparison_file)
df = df.sort_values("Test Accuracy", ascending=False)

best_accuracy = df.iloc[0]
best_loss = df.loc[df["Test Loss"].idxmin()]
best_gap = df.loc[df["Generalization Gap"].abs().idxmin()]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Top Accuracy Model", best_accuracy["Experiment"])
c2.metric("Highest Test Accuracy", f"{best_accuracy['Test Accuracy']:.2%}")
c3.metric("Lowest Test Loss", best_loss["Experiment"])
c4.metric("Closest Train–Validation Gap", best_gap["Experiment"])

st.subheader("🏆 Experiment Leaderboard")
st.dataframe(
    df.style.format({
        "Train Accuracy": "{:.2%}",
        "Validation Accuracy": "{:.2%}",
        "Test Accuracy": "{:.2%}",
        "Test Loss": "{:.4f}",
        "Best Validation Loss": "{:.4f}",
        "Generalization Gap": "{:.4%}"
    }),
    use_container_width=True,
    hide_index=True
)

left, right = st.columns(2)

with left:
    st.subheader("Test Accuracy")
    st.bar_chart(df.set_index("Experiment")["Test Accuracy"])

with right:
    st.subheader("Generalization Gap")
    gap_data = df.set_index("Experiment")["Generalization Gap"]
    st.bar_chart(gap_data)

st.divider()
st.subheader("🔬 Learning Curve Explorer")

history_files = {
    "Baseline": "baseline_history.csv",
    "Dropout": "dropout_history.csv",
    "L2": "l2_history.csv",
    "Early Stopping": "early_stopping_history.csv"
}

selected = st.selectbox("Select experiment", list(history_files.keys()))

history_path = RESULTS / history_files[selected]

if history_path.exists():
    history = pd.read_csv(history_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loss Curves**")
        st.line_chart(history[["loss", "val_loss"]])

    with col2:
        st.markdown("**Accuracy Curves**")
        st.line_chart(history[["accuracy", "val_accuracy"]])

    st.info(f"{selected} completed {len(history)} training epochs.")
else:
    st.warning("Learning history not found for this experiment.")

st.divider()
st.subheader("💡 Model Selection Insight")

st.write(
    f"**Accuracy leader:** {best_accuracy['Experiment']} "
    f"({best_accuracy['Test Accuracy']:.2%} test accuracy)"
)

st.write(
    f"**Lowest-loss model:** {best_loss['Experiment']} "
    f"({best_loss['Test Loss']:.4f} test loss)"
)

st.write(
    f"**Closest train–validation alignment:** {best_gap['Experiment']} "
    f"(gap {best_gap['Generalization Gap']:.4%})"
)

st.caption(
    "GeneralizeIQ | Validation, Regularization & Generalization Analysis"
)

st.divider()
st.subheader("🧬 Validation Intelligence")

selected_model = st.selectbox(
    "Analyze a model",
    df["Experiment"].tolist(),
    key="insight_model"
)

selected_row = df[df["Experiment"] == selected_model].iloc[0]

for title, explanation in analyze_model(selected_row):
    st.markdown(f"### {title}")
    st.write(explanation)
