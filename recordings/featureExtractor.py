import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy, linregress
from scipy.signal import find_peaks
from scipy.fft import fft
import math

# === Feature Calculation Functions (Same as Before) ===

def compute_time_domain_features(x, y, z):
    features = {}
    for axis, name in zip([x, y, z], ['x', 'y', 'z']):
        features[f'mean_{name}'] = np.mean(axis)
        features[f'median_{name}'] = np.median(axis)
        features[f'std_{name}'] = np.std(axis)
        features[f'var_{name}'] = np.var(axis)
        features[f'min_{name}'] = np.min(axis)
        features[f'max_{name}'] = np.max(axis)
        features[f'range_{name}'] = np.ptp(axis)
        features[f'rms_{name}'] = np.sqrt(np.mean(axis**2))
        features[f'energy_{name}'] = np.sum(axis**2) / len(axis)
        features[f'zero_crossing_rate_{name}'] = ((axis[:-1] * axis[1:]) < 0).sum()
        features[f'skewness_{name}'] = skew(axis)
        features[f'kurtosis_{name}'] = kurtosis(axis)
        features[f'peak_to_peak_{name}'] = np.max(axis) - np.min(axis)
    features['sma'] = np.mean(np.abs(x) + np.abs(y) + np.abs(z))
    return features

def compute_frequency_domain_features(x, y, z, sampling_rate=100):
    features = {}
    for axis, name in zip([x, y, z], ['x', 'y', 'z']):
        N = len(axis)
        freqs = np.fft.fftfreq(N, d=1/sampling_rate)
        fft_vals = np.abs(fft(axis))
        fft_vals = fft_vals[:N // 2]
        freqs = freqs[:N // 2]

        dominant_freq = freqs[np.argmax(fft_vals)]
        spectral_energy = np.sum(fft_vals**2) / N
        spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        normalized_spectrum = fft_vals / np.sum(fft_vals)
        spectral_entropy = entropy(normalized_spectrum)

        features[f'dominant_freq_{name}'] = dominant_freq
        features[f'spectral_energy_{name}'] = spectral_energy
        features[f'spectral_centroid_{name}'] = spectral_centroid
        features[f'spectral_entropy_{name}'] = spectral_entropy

        for i in range(5):
            features[f'fft_coeff_{name}_{i}'] = fft_vals[i]
    return features

def compute_magnitude_features(x, y, z):
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features = {
        'magnitude_mean': np.mean(magnitude),
        'magnitude_var': np.var(magnitude),
        'magnitude_rms': np.sqrt(np.mean(magnitude**2)),
    }
    return features

def compute_cross_axis_features(x, y, z):
    features = {
        'corr_xy': np.corrcoef(x, y)[0, 1],
        'corr_xz': np.corrcoef(x, z)[0, 1],
        'corr_yz': np.corrcoef(y, z)[0, 1],
        'cov_xy': np.cov(x, y)[0, 1],
        'cov_xz': np.cov(x, z)[0, 1],
        'cov_yz': np.cov(y, z)[0, 1],
    }
    return features

def compute_advanced_features(x, y, z):
    features = {}
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    features['svm'] = np.sum(magnitude) / len(magnitude)
    features['tilt_x'] = np.arccos(np.clip(x / magnitude, -1, 1)).mean()
    features['tilt_y'] = np.arccos(np.clip(y / magnitude, -1, 1)).mean()
    features['tilt_z'] = np.arccos(np.clip(z / magnitude, -1, 1)).mean()

    jerk_x = np.diff(x)
    jerk_y = np.diff(y)
    jerk_z = np.diff(z)
    jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
    features['jerk_mean'] = np.mean(jerk_magnitude)
    features['jerk_rms'] = np.sqrt(np.mean(jerk_magnitude**2))

    features['auc_x'] = np.trapz(np.abs(x))
    features['auc_y'] = np.trapz(np.abs(y))
    features['auc_z'] = np.trapz(np.abs(z))

    normalized_magnitude = magnitude / np.sum(magnitude)
    features['magnitude_entropy'] = entropy(normalized_magnitude)

    features['autocorr_x'] = np.corrcoef(x[:-1], x[1:])[0, 1]
    features['autocorr_y'] = np.corrcoef(y[:-1], y[1:])[0, 1]
    features['autocorr_z'] = np.corrcoef(z[:-1], z[1:])[0, 1]

    return features

def compute_windowed_features(x, y, z, threshold=0.5):
    features = {}
    for axis, name in zip([x, y, z], ['x', 'y', 'z']):
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(axis)), axis)
        features[f'slope_{name}'] = slope

        peaks, _ = find_peaks(axis)
        troughs, _ = find_peaks(-axis)
        features[f'num_peaks_{name}'] = len(peaks)
        features[f'num_troughs_{name}'] = len(troughs)

        features[f'activity_count_{name}'] = np.sum(np.abs(axis) > threshold)
    return features

def extract_features_from_window(window_df):
    x = window_df['acc_x'].values
    y = window_df['acc_y'].values
    z = window_df['acc_z'].values

    features = {}
    features.update(compute_time_domain_features(x, y, z))
    features.update(compute_frequency_domain_features(x, y, z))
    features.update(compute_magnitude_features(x, y, z))
    features.update(compute_cross_axis_features(x, y, z))
    features.update(compute_advanced_features(x, y, z))
    features.update(compute_windowed_features(x, y, z))
    
    return features

def extract_features_from_dataframe(df, window_size=100):
    feature_rows = []
    for start in range(0, len(df) - window_size + 1, window_size):
        window_df = df.iloc[start:start + window_size]
        features = extract_features_from_window(window_df)
        feature_rows.append(features)
    return pd.DataFrame(feature_rows)

# === Folder Traversal and Feature Extraction ===

def traverse_and_extract_features(root_folder, window_size=100, output_root="extracted_features"):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                filepath = os.path.join(dirpath, filename)
                print(f"Processing {filepath}...")

                try:
                    df = pd.read_csv(filepath)

                    # Check for correct columns
                    if not all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                        print(f"Skipping {filepath} (missing acc_x, acc_y, acc_z columns)")
                        continue

                    features_df = extract_features_from_dataframe(df, window_size=window_size)

                    relative_path = os.path.relpath(dirpath, root_folder)
                    save_dir = os.path.join(output_root, relative_path)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, f"features_{filename}")
                    features_df.to_csv(save_path, index=False)
                    print(f"Saved features to {save_path}")
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


# =============================================
# Run like this in your Jupyter Notebook:
# =============================================
# traverse_and_extract_features("path/to/your/folder", window_size=100)
