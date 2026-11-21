import pandas as pd
from pathlib import Path

# -------------------------------
# Change this path if necessary
# -------------------------------
protocol_file = Path(
    "dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")

# Read protocol file
df = pd.read_csv(
    protocol_file,
    sep=r"\s+",
    header=None,
    names=["speaker_id", "file_name", "unused", "attack", "label"])

# Keep only useful columns
df = df[["file_name", "label"]]

# Add .flac extension
df["file_name"] = df["file_name"] + ".flac"

print(df.head())

# Save CSV
df.to_csv("dataset/train_labels.csv", index=False)

print("\nTraining labels saved successfully!")