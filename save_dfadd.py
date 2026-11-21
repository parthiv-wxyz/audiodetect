from datasets import load_dataset
import soundfile as sf
import os

ds = load_dataset("SpeechAntiSpoofingBenchmarks/DFADD", split="test")

os.makedirs("dataset/dfadd/real", exist_ok=True)
os.makedirs("dataset/dfadd/fake", exist_ok=True)

real_count = 0
fake_count = 0

print("Saving DFADD samples...")

for item in ds:
    if item["label"] == 0 and real_count < 200:
        sf.write(
            f"dataset/dfadd/real/real_{real_count:04d}.wav",
            item["audio"]["array"],
            item["audio"]["sampling_rate"]
        )
        real_count += 1

    elif item["label"] == 1 and fake_count < 200:
        sf.write(
            f"dataset/dfadd/fake/fake_{fake_count:04d}.wav",
            item["audio"]["array"],
            item["audio"]["sampling_rate"]
        )
        fake_count += 1

    if real_count == 200 and fake_count == 200:
        break

print("Done!")
print("Real samples:", real_count)
print("Fake samples:", fake_count)