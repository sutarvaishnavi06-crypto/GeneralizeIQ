# GeneralizeIQ
## Neural Network Model Validation & Regularization Suite

GeneralizeIQ is an experimental suite for studying neural-network generalization on the Fashion-MNIST dataset.

### Dataset
Fashion-MNIST is loaded automatically through `tensorflow.keras.datasets`. No manual dataset download is required. The original 60,000 training images will later be divided into training and validation sets; the official 10,000-image test set will remain untouched until final evaluation.

### Planned experiments
1. Baseline neural network
2. Dropout regularization
3. L2 weight decay
4. Early stopping
5. Learning-curve and test-set comparison

### Environment
- Python 3.10 or 3.11 recommended
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas, Matplotlib, Scikit-learn

### Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

The application and experiments will be implemented and tested module by module.
