import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from pydub import AudioSegment

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_AUDIO_DIR = PROJECT_ROOT / "sample_audio"

OUTPUT_DIR = Path(__file__).resolve().parent / "audio_variations"

ORIGINAL_DIR = OUTPUT_DIR / "original"
NOISY_DIR = OUTPUT_DIR / "noisy"
OGG_DIR = OUTPUT_DIR / "ogg"
LOW_VOLUME_DIR = OUTPUT_DIR / "low_volume"


# Create folders
for folder in [
    ORIGINAL_DIR,
    NOISY_DIR,
    OGG_DIR,
    LOW_VOLUME_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# FILES FOR CASE STUDY 2
# ==========================================

TEST_FILES = [
    "LA_D_9944718.flac",
    "LA_D_9967770.flac",
    "LA_T_1138215.flac",
    "LA_T_1271820.flac",
    "LA_T_1272637.flac"
]


# ==========================================
# PROCESS EACH AUDIO
# ==========================================

for filename in TEST_FILES:

    input_path = SAMPLE_AUDIO_DIR / filename

    if not input_path.exists():
        print(f"❌ File not found: {filename}")
        continue

    print(f"\nProcessing: {filename}")

    # --------------------------------------
    # Load audio at 16 kHz
    # --------------------------------------

    audio, sr = librosa.load(
        input_path,
        sr=16000,
        mono=True
    )

    # ======================================
    # 1. ORIGINAL
    # ======================================

    original_path = ORIGINAL_DIR / filename

    sf.write(
        original_path,
        audio,
        16000
    )

    print("   ✓ Original")


    # ======================================
    # 2. BACKGROUND NOISE
    # ======================================

    # Generate controlled Gaussian noise
    noise = np.random.normal(
        0,
        0.005,
        len(audio)
    )

    noisy_audio = audio + noise

    # Prevent clipping
    noisy_audio = np.clip(
        noisy_audio,
        -1.0,
        1.0
    )

    noisy_filename = filename.replace(
        ".flac",
        "_noisy.wav"
    )

    noisy_path = NOISY_DIR / noisy_filename

    sf.write(
        noisy_path,
        noisy_audio,
        16000
    )

    print("   ✓ Noisy")


    # ======================================
    # 3. LOW VOLUME
    # ======================================

    # Reduce volume by approximately 10 dB
    low_volume_audio = audio * 0.316

    low_volume_audio = np.clip(
        low_volume_audio,
        -1.0,
        1.0
    )

    low_filename = filename.replace(
        ".flac",
        "_low_volume.wav"
    )

    low_path = LOW_VOLUME_DIR / low_filename

    sf.write(
        low_path,
        low_volume_audio,
        16000
    )

    print("   ✓ Low volume")


    # ======================================
    # 4. OGG COMPRESSION
    # ======================================

    # Convert original FLAC → WAV temporarily
    temp_wav = OUTPUT_DIR / f"temp_{filename}.wav"

    sf.write(
        temp_wav,
        audio,
        16000
    )

    audio_segment = AudioSegment.from_wav(
        temp_wav
    )

    ogg_filename = filename.replace(
        ".flac",
        ".ogg"
    )

    ogg_path = OGG_DIR / ogg_filename

    audio_segment.export(
        ogg_path,
        format="ogg",
        bitrate="64k"
    )

    # Remove temporary WAV
    temp_wav.unlink()

    print("   ✓ OGG compression")


print("\n" + "=" * 50)
print("CASE STUDY 2 AUDIO GENERATION COMPLETE")
print("=" * 50)

print(f"\nFiles saved inside:")
print(OUTPUT_DIR)
