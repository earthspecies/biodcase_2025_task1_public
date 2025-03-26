"""
Baseline system: do not produce any synchronization
Usage:
python baseline_nosync.py --output-dur=/path/to/desired/output/dir --inference-dir=/path/to/audio/folder
"""

import argparse
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    predictions = {"Filename": [], "Time Channel 0": [], "Time Channel 1": []}
    audio_fps = sorted(glob(os.path.join(args.inference_dir, "*")))

    for audio_fp in audio_fps:
        audio_dur = librosa.get_duration(path=audio_fp)
        keypoints = list(np.arange(0, audio_dur))
        fn = os.path.basename(audio_fp)
        predictions["Filename"].extend([fn for _ in keypoints])
        predictions["Time Channel 0"].extend(keypoints)
        predictions["Time Channel 1"].extend(keypoints)

    predictions = pd.DataFrame(predictions)
    output_fp = os.path.join(args.output_dir, f"predictions.csv")
    predictions.to_csv(output_fp, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    parser.add_argument("--inference-dir", type=str, required=True, help="path to audio folder to make predictions for")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
