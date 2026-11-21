try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

import os
import sys
import warnings
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt
import gradio as gr

from src.ensemble_detector import get_detector, SAMPLE_RATE

warnings.filterwarnings("ignore")

# ==========================================
# Settings & Dual-SSL Model Loading
# ==========================================
N_MFCC = 40
HOP_LENGTH = 512

detector = get_detector()


def generate_noise_reduced_preview(signal, sr=SAMPLE_RATE):
    """
    Produces a clean noise-reduced preview waveform (gentle high-pass + spectral gating)
    for playback only. Classification runs on the unmodified raw waveform.
    """
    try:
        # 1. Gentle 80 Hz high-pass filter to remove low-frequency rumble/hum
        nyq = 0.5 * sr
        b, a = butter(4, 80.0 / nyq, btype="high")
        filtered = filtfilt(b, a, signal).astype(np.float32)

        # 2. Mild spectral subtraction using lowest-energy frames as noise floor
        stft = librosa.stft(filtered, n_fft=512, hop_length=128)
        mag = np.abs(stft)
        phase = np.angle(stft)

        frame_energy = np.mean(mag ** 2, axis=0)
        noise_frames = max(1, int(len(frame_energy) * 0.15))
        quietest_idx = np.argsort(frame_energy)[:noise_frames]
        noise_profile = np.mean(mag[:, quietest_idx], axis=1, keepdims=True)

        # Soft attenuation mask (preserves speech naturalness)
        clean_mag = np.maximum(mag - 1.15 * noise_profile, 0.12 * mag)
        cleaned = librosa.istft(clean_mag * np.exp(1j * phase), hop_length=128, length=len(filtered))

        # Normalize to int16 for clean Gradio Audio playback
        peak = np.max(np.abs(cleaned)) + 1e-8
        if peak > 0.99:
            cleaned = cleaned / peak * 0.95
        audio_int16 = np.clip(cleaned * 32767.0, -32768, 32767).astype(np.int16)
        return (sr, audio_int16)
    except Exception:
        audio_int16 = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
        return (sr, audio_int16)


def plot_audio_visuals(signal, sr, mfcc):
    """Generates high-contrast Waveform and 40-Band MFCC Spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor("#0f172a")

    # 1. Waveform Plot
    times = np.linspace(0, len(signal) / sr, len(signal))
    axes[0].plot(times, signal, color="#38bdf8", lw=1.2)
    axes[0].set_title("Audio Waveform Dynamics", color="#f8fafc", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
    axes[0].tick_params(colors="#94a3b8", labelsize=8)
    axes[0].set_facecolor("#1e293b")
    axes[0].grid(True, color="#334155", linestyle="--", alpha=0.6)

    # 2. MFCC Heatmap
    img = axes[1].imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, len(signal) / sr, 0, N_MFCC]
    )
    axes[1].set_title("40-Band MFCC Spectrogram (Acoustic Feature Map)", color="#f8fafc", fontsize=11, fontweight="bold", pad=6)
    axes[1].set_xlabel("Timeline (Seconds)", color="#94a3b8", fontsize=9)
    axes[1].set_ylabel("MFCC Index", color="#94a3b8", fontsize=9)
    axes[1].tick_params(colors="#94a3b8", labelsize=8)
    axes[1].set_facecolor("#1e293b")

    cbar = fig.colorbar(img, ax=axes[1], pad=0.02)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.set_label("Energy (dB)", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig


@spaces.GPU(duration=60)
def analyze_audio_ui(audio_path):
    """Main processing function called by the Gradio Web Interface."""
    if audio_path is None:
        empty_msg = """
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 10px; padding: 20px; text-align: center; color: #94a3b8;">
            ⚠️ Please upload an audio file or record from microphone.
        </div>
        """
        return empty_msg, "", None, None, None, None

    try:
        res = detector.predict_file(audio_path)
    except Exception as e:
        err_msg = f"""
        <div style="background: rgba(127, 29, 29, 0.35); border: 1.5px solid #ef4444; border-radius: 10px; padding: 18px; text-align: center; color: #fca5a5;">
            ❌ Failed to analyze audio: {e}
        </div>
        """
        return err_msg, "", None, None, None, None

    if res.get("error"):
        err_msg = f"""
        <div style="background: rgba(127, 29, 29, 0.35); border: 1.5px solid #ef4444; border-radius: 10px; padding: 18px; text-align: center; color: #fca5a5;">
            ❌ {res['error']}
        </div>
        """
        return err_msg, "", None, None, None, None

    signal = res["signal"]
    sr = res["sr"]
    duration = res["duration"]
    is_spoof = res["is_spoof"]
    avg_spoof = res["avg_spoof"]
    bonafide_score = max(0.0, min(1.0, 1.0 - avg_spoof))
    segment_probs = res["segment_probs"]
    timestamps = res["timestamps"]
    spoof_threshold = res["spoof_threshold"]

    # Window duration (exact analyzed window span)
    if timestamps:
        window_dur = max(0.01, timestamps[0][1] - timestamps[0][0])
    else:
        window_dur = min(duration, 4.42)

    # Noise-reduced audio preview (playback only)
    denoised_preview = generate_noise_reduced_preview(signal, sr)

    # Extract MFCC for Acoustic Spectrogram Visualization
    full_mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    spectrogram_plot = plot_audio_visuals(signal, sr, full_mfcc)

    # Verdict Banner HTML (matching classic UI)
    if is_spoof:
        verdict_badge = f"""
        <div style="background: linear-gradient(135deg, rgba(127, 29, 29, 0.38) 0%, rgba(69, 10, 10, 0.45) 100%);
                    border: 1.5px solid #ef4444; border-radius: 10px; padding: 24px 20px; text-align: center;
                    box-shadow: 0 4px 18px rgba(239, 68, 68, 0.18);">
            <h2 style="color: #f87171; margin: 0; font-size: 22px; font-weight: 800; letter-spacing: 0.6px;">
                ⚠️ SPOOF DETECTED
            </h2>
            <p style="color: #e2e8f0; margin: 10px 0 4px; font-size: 14px; font-weight: 500;">
                Model spoof score: <b style="color: #f87171;">{avg_spoof * 100:.2f}%</b>
            </p>
            <p style="color: #94a3b8; margin: 0; font-size: 12.5px;">
                The model classified this audio as synthetic or spoof.
            </p>
        </div>
        """
    else:
        verdict_badge = f"""
        <div style="background: linear-gradient(135deg, rgba(20, 83, 45, 0.38) 0%, rgba(5, 46, 22, 0.45) 100%);
                    border: 1.5px solid #22c55e; border-radius: 10px; padding: 24px 20px; text-align: center;
                    box-shadow: 0 4px 18px rgba(34, 197, 94, 0.18);">
            <h2 style="color: #4ade80; margin: 0; font-size: 22px; font-weight: 800; letter-spacing: 0.6px;">
                ✅ BONAFIDE (AUTHENTIC HUMAN VOICE)
            </h2>
            <p style="color: #e2e8f0; margin: 10px 0 4px; font-size: 14px; font-weight: 500;">
                Model bonafide score: <b style="color: #4ade80;">{bonafide_score * 100:.2f}%</b>
            </p>
            <p style="color: #94a3b8; margin: 0; font-size: 12.5px;">
                The model classified this audio as authentic human speech.
            </p>
        </div>
        """

    # 5 KPI Stat Cards HTML (matching classic UI)
    stats_card = f"""
    <div style="display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px;">
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 8px; padding: 14px 8px; text-align: center;">
            <div style="color: #8b9bb4; font-size: 10.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.4px;">AUDIO DURATION</div>
            <div style="color: #f8fafc; font-size: 18px; font-weight: 700; margin-top: 4px;">{duration:.2f}s</div>
        </div>
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 8px; padding: 14px 8px; text-align: center;">
            <div style="color: #8b9bb4; font-size: 10.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.4px;">WINDOWS ANALYSED</div>
            <div style="color: #38bdf8; font-size: 18px; font-weight: 700; margin-top: 4px;">{len(segment_probs)}</div>
        </div>
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 8px; padding: 14px 8px; text-align: center;">
            <div style="color: #8b9bb4; font-size: 10.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.4px;">WINDOW DURATION</div>
            <div style="color: #f8fafc; font-size: 18px; font-weight: 700; margin-top: 4px;">{window_dur:.2f}s</div>
        </div>
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 8px; padding: 14px 8px; text-align: center;">
            <div style="color: #8b9bb4; font-size: 10.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.4px;">SPOOF SCORE</div>
            <div style="color: #f87171; font-size: 18px; font-weight: 700; margin-top: 4px;">{avg_spoof * 100:.2f}%</div>
        </div>
        <div style="background: #161f30; border: 1px solid #283548; border-radius: 8px; padding: 14px 8px; text-align: center;">
            <div style="color: #8b9bb4; font-size: 10.5px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.4px;">BONAFIDE SCORE</div>
            <div style="color: #4ade80; font-size: 18px; font-weight: 700; margin-top: 4px;">{bonafide_score * 100:.2f}%</div>
        </div>
    </div>
    """

    # Classification Score Distribution
    scores_dict = {
        "Authentic Human Voice (Bonafide)": round(bonafide_score, 4),
        "Synthetic / Spoof Voice": round(avg_spoof, 4)
    }

    # Segment Timeline Table
    timeline_data = []
    for (t_start, t_end), prob in zip(timestamps, segment_probs):
        status = "🔴 SPOOF" if prob >= spoof_threshold else "🟢 BONAFIDE"
        timeline_data.append([
            f"{t_start:.2f}s - {t_end:.2f}s",
            f"{prob * 100:.2f}%",
            status
        ])

    return verdict_badge, stats_card, denoised_preview, scores_dict, timeline_data, spectrogram_plot


# ==========================================
# Classic Dark-Slate Gradio UI Styling
# ==========================================
custom_css = """
:root, .dark {
    --body-background-fill: #0b0f19 !important;
    --background-fill-primary: #131b2b !important;
    --background-fill-secondary: #0f1624 !important;
    --block-background-fill: #131b2b !important;
    --block-border-color: #243045 !important;
    --border-color-primary: #243045 !important;
    --body-text-color: #e2e8f0 !important;
    --body-text-color-subdued: #94a3b8 !important;
    --block-label-background-fill: #2563eb !important;
    --block-label-text-color: #ffffff !important;
    --block-label-border-color: #2563eb !important;
    --block-title-text-color: #f8fafc !important;
    --button-primary-background-fill: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #3b82f6 !important;
}
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
}
#hero-banner {
    background: linear-gradient(135deg, #111827 0%, #172133 100%);
    border: 1px solid #2b3a55;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
}
.pill-badge {
    background-color: #0f172a;
    border: 1px solid #2563eb;
    color: #60a5fa;
    padding: 4px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
    display: inline-block;
}
#sys-info-box {
    background: transparent;
    padding: 8px 4px;
    color: #cbd5e1;
    font-size: 13.5px;
    line-height: 1.75;
}
#sys-info-box h3 {
    color: #f8fafc;
    font-size: 15px;
    font-weight: 700;
    margin: 6px 0 8px 0;
}
#sys-info-box ul {
    margin: 0;
    padding-left: 18px;
}
#sys-info-box li {
    margin-bottom: 2px;
    color: #cbd5e1;
}
#sys-info-box strong {
    color: #f1f5f9;
}
.section-heading {
    color: #f8fafc;
    font-size: 15px;
    font-weight: 700;
    margin: 14px 0 6px 2px;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="An Intelligent Speech Authentication System Using MFCC and CNN") as demo:
    gr.HTML("""
    <div id="hero-banner">
        <h1 style="color: #ffffff; margin: 0; font-size: 25px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
            🛡️ An Intelligent Speech Authentication System Using MFCC and CNN
        </h1>
        <p style="color: #94a3b8; margin: 6px 0 14px; font-size: 13.5px;">
            Audio Authenticity Detection Using MFCC and Convolutional Neural Network.
        </p>
        <div>
            <span class="pill-badge">ASVspoof 2019 LA</span>
            <span class="pill-badge">TensorFlow 2D CNN</span>
            <span class="pill-badge">40 MFCC Features</span>
            <span class="pill-badge">Binary Classification</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="📁 Upload Audio or Record Voice"
            )
            submit_btn = gr.Button("🔍 Run Audio Analysis", variant="primary", size="lg")

            gr.HTML("""
            <div id="sys-info-box">
                <h3>System Information</h3>
                <ul>
                    <li><strong>Model:</strong> 2D Convolutional Neural Network</li>
                    <li><strong>Feature extraction:</strong> MFCC</li>
                    <li><strong>MFCC coefficients:</strong> 40</li>
                    <li><strong>Input dimensions:</strong> 40 × 138</li>
                    <li><strong>Sample rate:</strong> 16 kHz</li>
                    <li><strong>Output:</strong> Bonafide / Spoof</li>
                </ul>
            </div>
            """)

        with gr.Column(scale=2):
            verdict_output = gr.HTML(label="Verdict Banner")
            stats_output = gr.HTML(label="Forensic Metrics")

            gr.HTML('<div class="section-heading">Noise Reduction Results</div>')
            denoised_audio = gr.Audio(
                label="Noise-Reduced Audio Preview",
                interactive=False
            )

            prob_label = gr.Label(
                label="Classification Score Distribution",
                num_top_classes=2
            )

            with gr.Tabs():
                with gr.TabItem("📊 Acoustic Waveform & Spectrogram"):
                    plot_output = gr.Plot(label="Acoustic Spectrogram")

                with gr.TabItem("⏱️ Tamper Timeline Breakdown"):
                    timeline_table = gr.Dataframe(
                        headers=["Time Window", "Spoof Score", "Verdict"],
                        datatype=["str", "str", "str"],
                        label="Window Analysis Breakdown"
                    )

    submit_btn.click(
        fn=analyze_audio_ui,
        inputs=audio_input,
        outputs=[verdict_output, stats_output, denoised_audio, prob_label, timeline_table, plot_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css=custom_css
    )
