---
title: Audio Deepfake Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🛡️ Deepfake Audio Detection using MFCC and 2D CNN

An industrial-grade acoustic forensic system capable of detecting synthetic speech, AI voice clones, and voice manipulation across arbitrary audio formats and durations.

---

## 🌟 Key Features
* **40-Band MFCC Extraction**: Captures subtle vocal tract anomalies and vocoder artifacts.
* **Dual-SSL Deep Architecture**: WavLM-Base + Wav2Vec2-XLSR backbones for state-of-the-art voice authenticity verification.
* **Universal Format & Length**: Supports `.wav`, `.mp3`, `.m4a`, `.ogg`, `.opus`, and `.flac` from 0.5s to 30+ minutes.
* **Adaptive Spectral Denoising**: Strips ambient background noise (traffic, fans, static) to prevent false alarms.
* **Temporal Tampering Localization**: Uses overlapping sliding windows to pinpoint the exact seconds of manipulation.
* **Interactive Cybersecurity Dashboard**: Built with Gradio featuring live microphone input, audio waveforms, and spectrogram visualizations.

---

## 🔬 Case Studies Addressed
1. **Generalization Test**: Out-of-distribution human microphone recordings verified with Cepstral Mean & Variance Normalization (CMVN).
2. **Codec Robustness**: Calibrated against lossy Opus/WhatsApp compression to prevent false positives.
3. **Noise Resilience**: Built-in Signal-to-Noise Ratio (SNR) profiling and spectral subtraction.
4. **Spliced Audio Detection**: Sliding-window timeline localization to catch short fake segments in long recordings.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
```
