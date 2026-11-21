import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==============================
# Create plots folder
# ==============================

os.makedirs("plots", exist_ok=True)

# ==============================
# Load Model
# ==============================

print("Loading trained model...")

model = load_model("models/cnn_audio_detector.keras")

print("Model Loaded Successfully!")

# ==============================
# Load Development Dataset
# ==============================

print("\nLoading development dataset...")

X_dev = np.load("processed/X_dev.npy")
y_dev = np.load("processed/y_dev.npy")

print("X_dev:", X_dev.shape)
print("y_dev:", y_dev.shape)

# ==============================
# Predictions
# ==============================

print("\nPredicting...")

y_prob = model.predict(X_dev)

y_pred = (y_prob > 0.5).astype(int).flatten()

# ==============================
# Metrics
# ==============================

accuracy = accuracy_score(y_dev, y_pred)
precision = precision_score(y_dev, y_pred)
recall = recall_score(y_dev, y_pred)
f1 = f1_score(y_dev, y_pred)
auc = roc_auc_score(y_dev, y_prob)

print("\n========== RESULTS ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

print("\nClassification Report\n")

print(classification_report(y_dev, y_pred))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_dev, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bonafide", "Spoof"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
plt.close()

# ==============================
# ROC Curve
# ==============================

fpr, tpr, _ = roc_curve(y_dev, y_prob)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("plots/roc_curve.png")
plt.close()

print("\nEvaluation Completed Successfully!")