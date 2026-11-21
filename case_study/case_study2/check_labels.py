import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
SAMPLE_AUDIO_DIR = PROJECT_ROOT / "sample_audio"
DEV_LABELS = PROJECT_ROOT / "dataset" / "dev_labels.csv"
TRAIN_LABELS = PROJECT_ROOT / "dataset" / "train_labels.csv"


# --------------------------------
# Load both label files
# --------------------------------

dev_labels = pd.read_csv(
    DEV_LABELS,
    header=None,
    names=["filename", "label"]
)

train_labels = pd.read_csv(
    TRAIN_LABELS,
    header=None,
    names=["filename", "label"]
)


# --------------------------------
# Combine labels
# --------------------------------

all_labels = pd.concat(
    [dev_labels, train_labels],
    ignore_index=True
)

label_dict = dict(
    zip(all_labels["filename"], all_labels["label"])
)


# --------------------------------
# Check sample audio files
# --------------------------------

print("\nCase Study 2 - Audio Labels")
print("=" * 65)

for audio_file in sorted(SAMPLE_AUDIO_DIR.glob("*.flac")):

    filename = audio_file.name

    if filename in label_dict:

        actual_label = label_dict[filename]

        print(
            f"{filename:<35} -> {actual_label}"
        )

    else:

        print(
            f"{filename:<35} -> NOT FOUND"
        )