import pandas as pd

protocol_file = "dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

rows = []

with open(protocol_file, "r") as f:
    for line in f:
        parts = line.strip().split()

        filename = parts[1] + ".flac"
        label = parts[-1]

        rows.append([filename, label])

df = pd.DataFrame(rows, columns=["file_name", "label"])

print(df.head())

df.to_csv("dataset/dev_labels.csv", index=False)

print("\nSaved Successfully!")
print("Total Files:", len(df))