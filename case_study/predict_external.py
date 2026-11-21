import os
import librosa
import numpy as np
import tensorflow as tf

# -------------------------
# Configuration
# -------------------------

MODEL_PATH = "models/cnn_audio_detector.keras"     # Change if your model is elsewhere
AUDIO_PATH = "case_study/real_audio/midhun.wav"

MAX_FRAMES = 138
SAMPLE_RATE = 16000
N_MFCC = 40

# -------------------------
# Load Model
# -------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Pad / Truncate
# -------------------------

def pad_or_truncate(mfcc):

    current_frames = mfcc.shape[1]

    if current_frames < MAX_FRAMES:

        pad_width = MAX_FRAMES - current_frames

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc


# -------------------------
# Feature Extraction
# -------------------------

signal, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=N_MFCC
)

mfcc = pad_or_truncate(mfcc)

mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

mfcc = np.expand_dims(mfcc, axis=-1)

mfcc = np.expand_dims(mfcc, axis=0)

# -------------------------
# Prediction
# -------------------------

prediction = model.predict(mfcc, verbose=0)[0][0]

print("Raw Output :", prediction)

if prediction >= 0.5:
    print("Prediction : Fake")
    print(f"Confidence : {prediction*100:.2f}%")
else:
    print("Prediction : Real")
    print(f"Confidence : {(1-prediction)*100:.2f}%")