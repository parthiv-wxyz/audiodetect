import os
import sys
import glob
import warnings

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure sibling imports work whether invoked from repo root or src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ensemble_detector import (
    DualSSLDeepfakeDetector,
    get_detector,
    load_audio_universal,
    SAMPLE_RATE,
)

warnings.filterwarnings("ignore")


def detect_deepfake(audio_path, detector=None):
    """
    Evaluates audio of ANY format and ANY duration using the Dual-SSL Deepfake Ensemble
    (WavLM-Base + Wav2Vec2-Large-XLSR).
    """
    if detector is None:
        detector = get_detector()

    res = detector.predict_file(audio_path)

    if res.get("error"):
        return {
            "file": os.path.basename(audio_path),
            "duration": f"{res.get('duration', 0.0):.2f}s",
            "verdict": f"ERROR ({res['error']})",
            "confidence": "0.0%",
            "avg_spoof_prob": "0.0%",
            "max_spoof_prob": "0.0%",
            "wavlm_spoof_prob": "0.0%",
            "xlsr_spoof_prob": "0.0%",
            "fake_segments_count": 0,
            "tampered_intervals": [],
        }

    fake_segments = []
    thresh = res["spoof_threshold"]
    for (t_start, t_end), prob in zip(res["timestamps"], res["segment_probs"]):
        if prob >= thresh:
            fake_segments.append({
                "start": f"{int(t_start // 60):02d}:{t_start % 60:05.2f}",
                "end": f"{int(t_end // 60):02d}:{t_end % 60:05.2f}",
                "spoof_prob": round(prob * 100, 2),
            })

    return {
        "file": res["file"],
        "duration": f"{res['duration']:.2f}s",
        "verdict": res["verdict"],
        "confidence": f"{res['confidence'] * 100:.2f}%",
        "avg_spoof_prob": f"{res['avg_spoof'] * 100:.2f}%",
        "max_spoof_prob": f"{res['max_spoof'] * 100:.2f}%",
        "wavlm_spoof_prob": f"{res['wavlm_avg'] * 100:.2f}%",
        "xlsr_spoof_prob": f"{res['xlsr_avg'] * 100:.2f}%",
        "fake_segments_count": len(fake_segments),
        "tampered_intervals": fake_segments,
    }


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python src/universal_detector.py test/sample.mp3")
        print("  python src/universal_detector.py test/\n")
        sys.exit(1)

    target_path = sys.argv[1]
    detector = get_detector()

    if os.path.isdir(target_path):
        exts = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.opus", "*.flac", "*.aac", "*.mp4", "*.mkv", "*.webm")
        audio_files = []
        for ext in exts:
            audio_files.extend(glob.glob(os.path.join(target_path, ext)))
        if not audio_files:
            print(f"No audio files found in: {target_path}")
            sys.exit(1)
    else:
        audio_files = [target_path]

    print(f"\nFound {len(audio_files)} file(s) to inspect.\n" + "=" * 68)

    for fpath in audio_files:
        res = detect_deepfake(fpath, detector)

        print(f"Audio File  : {res['file']} (Duration: {res['duration']})")
        print(f"Verdict     : {res['verdict']}")
        print(f"Confidence  : {res['confidence']}")
        print(f"Ensemble %  : Avg {res['avg_spoof_prob']} | Peak {res['max_spoof_prob']}")
        print(f"Backbones   : WavLM-Base {res['wavlm_spoof_prob']} | Wav2Vec2-XLSR {res['xlsr_spoof_prob']}")

        if res["fake_segments_count"] > 0:
            print(f"\n[!] ALERT: {res['fake_segments_count']} Manipulated Segment(s) Detected:")
            for item in res["tampered_intervals"]:
                print(f"    - {item['start']} to {item['end']} --> Spoof Probability: {item['spoof_prob']}%")
        else:
            print("Status      : Clean (No synthetic manipulation detected)")

        print("=" * 68)


if __name__ == "__main__":
    main()
