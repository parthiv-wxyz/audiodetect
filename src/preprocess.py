import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ==============================
# Paths & Settings
# ==============================
LABELS_CSV = "dataset/train_labels.csv"
AUDIO_FOLDER = "dataset/LA/ASVspoof2019_LA_train/flac"
OUTPUT_FOLDER = "processed"

N_MFCC = 40
MAX_FRAMES = 138
SAMPLE_RATE = 16000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_labels():
    if os.path.exists(LABELS_CSV):
        labels = pd.read_csv(LABELS_CSV)
        print(labels.head())
        print("\nTotal Files:", len(labels))
        return labels
    return None


def label_to_number(label):
    return 0 if label.strip().lower() == "bonafide" else 1


def pad_or_tile(mfcc, target_frames=MAX_FRAMES):
    """
    If audio is shorter than target_frames, tile/repeat it so the window
    is filled with speech rather than artificial dead zeros.
    If longer, truncate to target_frames.
    """
    current_frames = mfcc.shape[1]
    if current_frames < target_frames:
        repeats = int(np.ceil(target_frames / current_frames))
        mfcc = np.tile(mfcc, (1, repeats))[:, :target_frames]
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc


def extract_features(audio_path):
    """
    Loads audio, extracts MFCCs, applies CMVN normalization on speech frames,
    and formats to fixed frame length.
    """
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Extract MFCC on active speech
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    
    # 1. Apply CMVN (Cepstral Mean & Variance Normalization) per frequency band
    # This cancels out microphone frequency biases and room EQ
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mean) / (std + 1e-8)
    
    # 2. Tile or truncate to target frame length AFTER normalization
    mfcc = pad_or_tile(mfcc, target_frames=MAX_FRAMES)
    
    return mfcc


def main():
    labels = load_labels()
    if labels is None:
        print(f"Labels CSV '{LABELS_CSV}' not found.")
        return

    X = []
    y = []

    print("Extracting features from training audio...")
    for index, row in tqdm(labels.iterrows(), total=len(labels)):
        audio_path = os.path.join(AUDIO_FOLDER, row["file_name"])
        
        if not os.path.exists(audio_path):
            continue
            
        mfcc_feat = extract_features(audio_path)
        X.append(mfcc_feat)
        y.append(label_to_number(row["label"]))

    print("\nProcessed Files:", len(X))
    print("Label Distribution:", Counter(y))

    if len(X) == 0:
        print("No audio files found in AUDIO_FOLDER.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = np.expand_dims(X, axis=-1)

    print("\nFinal Shapes:")
    print("X :", X.shape)
    print("y :", y.shape)

    np.save(os.path.join(OUTPUT_FOLDER, "X_train.npy"), X)
    np.save(os.path.join(OUTPUT_FOLDER, "y_train.npy"), y)
    print("\nSaved training features successfully!")

if __name__ == "__main__":
    main()