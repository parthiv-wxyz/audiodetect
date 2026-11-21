import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from cnn_model import build_cnn

# ==============================
# Create folders
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ==============================
# Load Dataset
# ==============================
print("Loading dataset...")

X_train = np.load("processed/X_train.npy")
y_train = np.load("processed/y_train.npy")

X_dev = np.load("processed/X_dev.npy")
y_dev = np.load("processed/y_dev.npy")

print("\nTraining Data")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nDevelopment Data")
print("X_dev:", X_dev.shape)
print("y_dev:", y_dev.shape)

# ==============================
# Class Weights (handles 90/10 imbalance)
# ==============================
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights (0 = Bonafide/Real, 1 = Spoof/Fake):")
print(class_weights)

# ==============================
# Build Model
# ==============================
model = build_cnn()

# ==============================
# Compile with Multi-Metric Evaluation
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\nModel Compiled Successfully!")
model.summary()

# ==============================
# Callbacks
# ==============================
early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/cnn_audio_detector.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# Train
# ==============================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_dev, y_dev),
    epochs=25,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# ==============================
# Save Training Curves
# ==============================
print("\nSaving training performance plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Binary Crossentropy Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history.history["accuracy"], label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

# AUC-ROC
axes[2].plot(history.history["auc"], label="Train AUC")
axes[2].plot(history.history["val_auc"], label="Val AUC")
axes[2].set_title("AUC-ROC Curve Metric")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=300)
print("Saved learning curves to plots/training_curves.png")

# ==============================
# Evaluation on Development Set
# ==============================
print("\nEvaluating best model on Development Set (X_dev)...")
y_pred_probs = model.predict(X_dev, batch_size=32)
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

print("\nClassification Report (Dev Set):")
print(classification_report(y_dev, y_pred, target_names=["Bonafide (Real)", "Spoof (Fake)"]))

cm = confusion_matrix(y_dev, y_pred)
print("Confusion Matrix:")
print("                 Predicted Real   Predicted Fake")
print(f"Actual Real    :      {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Fake    :      {cm[1][0]:<14} {cm[1][1]}")

auc = roc_auc_score(y_dev, y_pred_probs)
print(f"\nFinal Dev Set AUC-ROC: {auc * 100:.2f}%")