import os
import pandas as pd
from scipy.stats import zscore

# === Preprocessing Functions ===

def leaky_integrator(signal, lambda_=0.9):
    smoothed = [signal.iloc[0]]
    for i in range(1, len(signal)):
        smoothed.append(lambda_ * smoothed[i - 1] + (1 - lambda_) * signal.iloc[i])
    return smoothed

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the relevant accelerometer columns
    acc = df[["acc_x", "acc_y", "acc_z"]].copy()

    # Apply leaky integrator smoothing
    acc["x_smooth"] = leaky_integrator(acc["acc_x"])
    acc["y_smooth"] = leaky_integrator(acc["acc_y"])
    acc["z_smooth"] = leaky_integrator(acc["acc_z"])

    # Apply z-score normalization
    acc["x_norm"] = zscore(acc["x_smooth"])
    acc["y_norm"] = zscore(acc["y_smooth"])
    acc["z_norm"] = zscore(acc["z_smooth"])

    # Return only normalized columns
    return acc[["x_norm", "y_norm", "z_norm"]]

# === Folder Setup ===

BASE_DIR = "Dataset"
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed_data")

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Iterate Over Character Folders and CSV Files ===

for char_folder in os.listdir(BASE_DIR):
    char_path = os.path.join(BASE_DIR, char_folder)

    # Skip non-folders and the output folder itself
    if not os.path.isdir(char_path) or char_folder == "preprocessed_data":
        continue

    # Create output subfolder for the character
    output_char_folder = os.path.join(OUTPUT_DIR, char_folder)
    os.makedirs(output_char_folder, exist_ok=True)

    for csv_file in os.listdir(char_path):
        if not csv_file.endswith(".csv"):
            continue

        input_path = os.path.join(char_path, csv_file)

        try:
            df = pd.read_csv(input_path)
            processed_df = preprocess(df)

            # Prepare new filename
            file_root, ext = os.path.splitext(csv_file)
            new_filename = f"{file_root}_preprocessed{ext}"
            output_path = os.path.join(output_char_folder, new_filename)

            # Save processed file
            processed_df.to_csv(output_path, index=False)
            print(f"[✓] Processed: {input_path} -> {output_path}")

        except Exception as e:
            print(f"[✗] Failed to process {input_path}: {e}")
