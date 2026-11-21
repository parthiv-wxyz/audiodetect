import os
import sys
import warnings
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
WAVLM_MODEL_ID = "DavidCombei/wavLM-base-Deepfake_V2"
XLSR_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
COMPRESSED_EXTS = {".ogg", ".opus", ".webm"}


def load_audio_universal(audio_path, target_sr=SAMPLE_RATE):
    """Robust audio loader supporting .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .webm."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    try:
        signal, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if len(signal) > 0:
            return signal.astype(np.float32), sr
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        signal = samples / (2 ** (audio.sample_width * 8 - 1))
        return signal.astype(np.float32), target_sr
    except Exception as e:
        raise RuntimeError(f"Could not decode audio '{audio_path}': {e}")


def make_windows(signal, sr=SAMPLE_RATE, win_sec=5.0, hop_sec=4.0, max_windows=6):
    """Splits audio into evaluation windows while preserving full context for short clips."""
    duration = len(signal) / sr
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)

    if len(signal) <= int(6.5 * sr):
        return [(0.0, round(duration, 2), signal)]

    windows = []
    for start in range(0, len(signal) - win_samples + 1, hop_samples):
        t_start = round(start / sr, 2)
        t_end = round(min((start + win_samples) / sr, duration), 2)
        windows.append((t_start, t_end, signal[start : start + win_samples]))

    if (len(signal) - win_samples) % hop_samples != 0:
        start = len(signal) - win_samples
        t_start = round(start / sr, 2)
        windows.append((t_start, round(duration, 2), signal[start:]))

    if len(windows) > max_windows:
        idxs = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        windows = [windows[i] for i in idxs]

    return windows


class DualSSLDeepfakeDetector:
    """
    Dual-Backbone Self-Supervised Learning (SSL) Audio Deepfake Detector:
      - WavLM-Base (DavidCombei/wavLM-base-Deepfake_V2): catches ASVspoof, neural vocoder,
        FLAC, noisy, and silence-padded synthetic speech.
      - Wav2Vec2-Large-XLSR (garystafford/wav2vec2-deepfake-voice-detector): catches modern
        commercial TTS / MP3 voice clones and suppresses false positives on real/compressed speech.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM-Base backbone ({WAVLM_MODEL_ID}) on {self.device}...")
        self.fe_wavlm = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
        self.m_wavlm = AutoModelForAudioClassification.from_pretrained(WAVLM_MODEL_ID).to(self.device).eval()

        print(f"Loading Wav2Vec2-XLSR backbone ({XLSR_MODEL_ID}) on {self.device}...")
        self.fe_xlsr = AutoFeatureExtractor.from_pretrained(XLSR_MODEL_ID)
        self.m_xlsr = AutoModelForAudioClassification.from_pretrained(XLSR_MODEL_ID).to(self.device).eval()
        print("Dual-SSL Deepfake Detection Engine Loaded Successfully!")

    def predict_signal(self, signal, sr=SAMPLE_RATE, audio_path=""):
        duration = len(signal) / sr
        if duration < 0.3:
            return {
                "error": "Audio too short. Please provide at least 0.5s of speech.",
                "duration": duration,
            }

        windows = make_windows(signal, sr=sr)
        timestamps = []
        wavlm_probs = []
        xlsr_probs = []

        # Ensure model is on CUDA if dynamically allocated by ZeroGPU
        current_device = "cuda" if torch.cuda.is_available() else self.device
        if current_device != self.device:
            self.device = current_device
            self.m_wavlm = self.m_wavlm.to(self.device)
            self.m_xlsr = self.m_xlsr.to(self.device)

        with torch.no_grad():
            for t_start, t_end, chunk in windows:
                timestamps.append((t_start, t_end))

                # WavLM-Base: LABEL_0 = Fake/Spoof, LABEL_1 = Real/Bonafide
                in_w = self.fe_wavlm(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                in_w = {k: v.to(self.device) for k, v in in_w.items()}
                logits_w = self.m_wavlm(**in_w).logits
                p_w = float(torch.softmax(logits_w, dim=-1)[0, 0].item())
                wavlm_probs.append(p_w)

                # Wav2Vec2-XLSR: LABEL_0 = Real, LABEL_1 = Fake
                in_x = self.fe_xlsr(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                in_x = {k: v.to(self.device) for k, v in in_x.items()}
                logits_x = self.m_xlsr(**in_x).logits
                p_x = float(torch.softmax(logits_x, dim=-1)[0, 1].item())
                xlsr_probs.append(p_x)

        wavlm_avg = float(np.mean(wavlm_probs))
        xlsr_avg = float(np.mean(xlsr_probs))

        ext = os.path.splitext(audio_path)[1].lower() if audio_path else ""
        is_compressed = ext in COMPRESSED_EXTS

        # Calibrated decision thresholds:
        # On lossy compressed voice notes (.ogg/.opus/.webm) where XLSR strongly confirms human (<5%),
        # require higher WavLM consensus (80%) to prevent single-frame codec artifacts from false-flagging.
        wavlm_thresh = 0.80 if (is_compressed and xlsr_avg < 0.05) else 0.65
        xlsr_thresh = 0.50

        is_spoof = (wavlm_avg >= wavlm_thresh) or (xlsr_avg >= xlsr_thresh)

        # Compute per-window fused scores
        segment_probs = []
        for p_w, p_x in zip(wavlm_probs, xlsr_probs):
            if is_compressed and p_x < 0.05 and wavlm_avg < 0.80:
                fused = 0.35 * p_w + 0.65 * p_x
            else:
                fused = max(p_w, p_x)
            segment_probs.append(float(fused))

        if is_spoof:
            overall_spoof_prob = max(wavlm_avg, xlsr_avg)
            overall_confidence = overall_spoof_prob
        else:
            if is_compressed and xlsr_avg < 0.05 and wavlm_avg < 0.80:
                overall_spoof_prob = 0.35 * wavlm_avg + 0.65 * xlsr_avg
            else:
                overall_spoof_prob = max(wavlm_avg, xlsr_avg)
            overall_confidence = 1.0 - overall_spoof_prob

        peak_spoof = float(np.max(segment_probs))
        avg_spoof = float(overall_spoof_prob)

        return {
            "error": None,
            "duration": duration,
            "is_spoof": is_spoof,
            "verdict": "SPOOF (Fake / Manipulated Voice)" if is_spoof else "BONAFIDE (Real Human Voice)",
            "confidence": overall_confidence,
            "avg_spoof": avg_spoof,
            "max_spoof": peak_spoof,
            "wavlm_avg": wavlm_avg,
            "xlsr_avg": xlsr_avg,
            "wavlm_probs": wavlm_probs,
            "xlsr_probs": xlsr_probs,
            "segment_probs": segment_probs,
            "timestamps": timestamps,
            "spoof_threshold": 0.50,
        }

    def predict_file(self, audio_path):
        signal, sr = load_audio_universal(audio_path, target_sr=SAMPLE_RATE)
        result = self.predict_signal(signal, sr=sr, audio_path=audio_path)
        result["signal"] = signal
        result["sr"] = sr
        result["file"] = os.path.basename(audio_path)
        return result


_DETECTOR_SINGLETON = None


def get_detector():
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        _DETECTOR_SINGLETON = DualSSLDeepfakeDetector()
    return _DETECTOR_SINGLETON
