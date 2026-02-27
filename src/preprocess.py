import glob
import os
from pathlib import Path

import cv2
import numpy as np
from pymatreader import read_mat
from scipy.signal import butter, filtfilt, spectrogram
from tqdm import tqdm

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw"
PROCESSED_PATH = REPO_ROOT / "data" / "processed"
FS = 128  # Sampling frequency
WINDOW_SEC = 5

# Create output directories
CLASSES = ["awake", "light", "deep"]
for c in CLASSES:
    (PROCESSED_PATH / c).mkdir(parents=True, exist_ok=True)


def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=128, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def process_files():
    mat_files = glob.glob(str(RAW_DATA_PATH / "*.mat"))
    print(f"Found {len(mat_files)} files in {RAW_DATA_PATH}")

    img_count = 0

    for i, file_path in enumerate(tqdm(mat_files)):
        try:
            mat = read_mat(file_path)

            # --- EXTRACT DATA (Updated for your specific file structure) ---

            # 1. Find EEG (Looking for 'EEG', 'eeg_signal', etc.)
            if "EEG" in mat:
                eeg = mat["EEG"]  # <--- THIS IS THE FIX
            elif "eeg_signal" in mat:
                eeg = mat["eeg_signal"]
            elif "vals" in mat:
                eeg = mat["vals"][0]
            elif "data" in mat:
                eeg = mat["data"]
            else:
                print(f"\nSkipping {os.path.basename(file_path)}: Keys: {mat.keys()}")
                continue

            # 2. Find BIS
            if "bis" in mat:
                bis = mat["bis"]
            elif "BIS" in mat:
                bis = mat["BIS"]
            else:
                bis = np.zeros(len(eeg))

            # Ensure 1D arrays and float type
            eeg = np.array(eeg).flatten().astype(float)
            bis = np.array(bis).flatten().astype(float)

            # Filter EEG
            eeg_clean = butter_bandpass_filter(eeg, fs=FS)

            # Slide window and generate spectrograms
            window_samples = WINDOW_SEC * FS
            for start in range(0, len(eeg_clean) - window_samples, window_samples):
                segment = eeg_clean[start : start + window_samples]
                f, t, Sxx = spectrogram(segment, fs=FS, nperseg=256, noverlap=128)

                # Convert to dB scale
                Sxx_db = 10 * np.log10(Sxx + 1e-8)

                # Normalize to [0, 255] for image saving
                Sxx_norm = cv2.normalize(Sxx_db, None, 0, 255, cv2.NORM_MINMAX)
                Sxx_uint8 = Sxx_norm.astype("uint8")

                # Save image
                output_class = "awake" if np.mean(bis) < 40 else "deep" if np.mean(bis) > 60 else "light"
                output_dir = PROCESSED_PATH / output_class
                filename = f"{Path(file_path).stem}_win{start // window_samples}.png"
                cv2.imwrite(str(output_dir / filename), Sxx_uint8)
                img_count += 1

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

    print(f"\n✅ Finished processing. Total images saved: {img_count}")


if __name__ == "__main__":
    process_files()
