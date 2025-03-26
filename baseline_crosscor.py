"""
Baseline system: Assume drift is affine (time scaling + offset). Use spectral cross-correlation to determine which of the candidate drifts is the best.
Usage:
python baseline_crosscor.py --output-dur=/path/to/desired/output/dir --inference-dir=/path/to/audio/folder
"""

import argparse
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import generate_candidate_keypoints, remove_duplicate_keypoints


def window_and_compute_similarity(i, keypoints_0, keypoints_1, sr, window_size, audio):
    """
    Extracts audio segments based on keypoints and computes the dot-product similarity of their mel spectrograms.

    Args:
    i (int): Index of the keypoint to process.
    keypoints_0 (np.ndarray): A 1D NumPy array of keypoints for channel 0.
    keypoints_1 (np.ndarray): A 1D NumPy array of keypoints for channel 1.
    sr (int): Sample rate of the audio.
    window_size (float): Duration in seconds of the window used for similarity computation.
    audio (np.ndarray): A 2D NumPy array of shape (2, n_samples) containing two-channel audio data.

    Returns:
    float: The computed similarity score between the extracted audio segments.
    """

    k0 = int(sr * keypoints_0[i])
    k1 = int(sr * keypoints_1[i])
    dur = int(sr * window_size)
    audio_0 = audio[0, 0 : k0 + dur]
    audio_1 = audio[1, k1 : k1 + dur]

    m = min(len(audio_0), len(audio_1))
    if m == 0:
        return 0

    audio_0 = audio_0[:m]
    audio_1 = audio_1[:m]

    spec_0 = librosa.feature.melspectrogram(y=audio_0, sr=sr)
    spec_1 = librosa.feature.melspectrogram(y=audio_1, sr=sr)

    similarity = np.mean(spec_0 * spec_1)
    return similarity


def compute_similarity(audio, sr, keypoints_0, keypoints_1, window_size):
    """
    Compute the average dot product similarity between mel spectrograms of two audio channels.

    Args:
    audio (np.ndarray): A 2D NumPy array of shape (2, n_samples) containing two-channel audio data.
    sr (int): Sample rate of the audio.
    keypoints_0 (np.ndarray): A 1D NumPy array of keypoints for channel 0.
    keypoints_1 (np.ndarray): A 1D NumPy array of keypoints for channel 1.
    window_size (float): Duration in seconds of the window used for similarity computation.

    Returns:
    float: The average similarity score between the two audio channels.
    """

    keypoints_0, keypoints_1 = remove_duplicate_keypoints(keypoints_0, keypoints_1)
    similarity = []

    n_jobs = -1  # Use all available cores
    similarity = Parallel(n_jobs=n_jobs)(
        delayed(window_and_compute_similarity)(i, keypoints_0, keypoints_1, sr, window_size, audio)
        for i in range(len(keypoints_0))
    )

    similarity = np.mean(similarity)
    return similarity


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    predictions = {"Filename": [], "Time Channel 0": [], "Time Channel 1": []}
    audio_fps = sorted(glob(os.path.join(args.inference_dir, "*")))

    for audio_fp in tqdm(audio_fps):
        audio, sr = librosa.load(audio_fp, mono=False, sr=None)
        audio_dur = librosa.get_duration(path=audio_fp)

        keypoints_0 = np.arange(0, audio_dur)
        candidate_keypoints_1 = generate_candidate_keypoints(
            keypoints_0, args.max_error, audio_dur, args.n_delays_to_try
        )

        best_keypoints_1 = keypoints_0
        best_similarity = -np.infty

        for keypoints_1 in candidate_keypoints_1:
            similarity = compute_similarity(audio, sr, keypoints_0, keypoints_1, args.window_size)
            if similarity > best_similarity:
                best_similarity = similarity
                best_keypoints_1 = keypoints_1

        fn = os.path.basename(audio_fp)
        keypoints_0 = list(keypoints_0)
        keypoints_1 = list(best_keypoints_1)
        predictions["Filename"].extend([fn for _ in keypoints_0])
        predictions["Time Channel 0"].extend(keypoints_0)
        predictions["Time Channel 1"].extend(keypoints_1)

    predictions = pd.DataFrame(predictions)
    output_fp = os.path.join(args.output_dir, f"predictions.csv")
    predictions.to_csv(output_fp, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    parser.add_argument("--inference-dir", type=str, required=True, help="path to audio folder to make predictions for")
    parser.add_argument(
        "--max-error", type=float, default=5.0, help="max difference between Time Channel 0 and Time Channel 1"
    )
    parser.add_argument(
        "--window-size", type=float, default=1.0, help="duration of window size used to compute similarity"
    )
    parser.add_argument(
        "--n-delays-to-try",
        type=int,
        default=10,
        help="controls how many candidate keypoint sets to try before choosing the best. Run time is quadratic in this number",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
