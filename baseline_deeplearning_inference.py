"""
Baseline system: Assume drift is affine (time scaling + offset). Use learned representation to determine which of the candidate drifts is the best.
Usage:
python baseline_deeplearning_inference.py --output-dur=/path/to/desired/output/dir --inference-dir=/path/to/audio/folder --pretrained-fp=/path/to/model.pt
"""

import argparse
import os
from glob import glob

import librosa
import numpy as np
import pandas as pd
import torch
from models import BEATsEncoderAndMLP
from tqdm import tqdm
from utils import generate_candidate_keypoints, load_audio, pad_to_dur, remove_duplicate_keypoints

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_similarity(audio, sr, keypoints_0, keypoints_1, window_size, model):
    """
    Compute the similarity between two audio channels using a trained model.

    Args:
    audio (np.ndarray): A 2D NumPy array of shape (2, n_samples) containing two-channel audio data.
    sr (int): Sample rate of the audio.
    keypoints_0 (np.ndarray): A 1D NumPy array of keypoints for channel 0.
    keypoints_1 (np.ndarray): A 1D NumPy array of keypoints for channel 1.
    window_size (float): Duration in seconds of the window over which similarity is computed.
    model (torch.nn.Module): A model that computes similarity between audio segments.

    Returns:
    float: The computed similarity score between the two audio channels. Returns -infinity if similarity is too low.
    """

    keypoints_0, keypoints_1 = remove_duplicate_keypoints(keypoints_0, keypoints_1)

    # compute similarity
    similarity = []
    audio_0s = []
    audio_1s = []

    for i in range(len(keypoints_0)):
        k0 = int(sr * keypoints_0[i])
        k1 = int(sr * keypoints_1[i])
        dur = int(sr * window_size)
        audio_0 = audio[0, k0 : k0 + dur]
        audio_1 = audio[1, k1 : k1 + dur]
        audio_0 = pad_to_dur(audio_0, sr, window_size)
        audio_1 = pad_to_dur(audio_1, sr, window_size)

        audio_0s.append(audio_0)
        audio_1s.append(audio_1)

    if len(audio_0s) < 2:
        return -np.infty

    audio_0s = torch.stack(audio_0s)
    audio_1s = torch.stack(audio_1s)

    dataset = torch.utils.data.TensorDataset(audio_0s, audio_1s)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False
    )

    for audio_0, audio_1 in dataloader:
        with torch.no_grad():
            sim = model(audio_0, audio_1).cpu().numpy()
        similarity.append(sim)

    similarity = np.concatenate(similarity)
    similarity = np.sum(similarity > 0)
    if similarity < 1:
        similarity = -np.infty

    return similarity


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    predictions = {"Filename": [], "Time Channel 0": [], "Time Channel 1": []}
    audio_fps = sorted(glob(os.path.join(args.inference_dir, "*")))

    # Load pretrained model
    model = BEATsEncoderAndMLP(args.beats_fp).to(device)
    model.load_state_dict(torch.load(args.pretrained_fp, weights_only=True))
    model.eval()

    # Iterate through audio files
    for audio_fp in tqdm(audio_fps):
        audio, sr = load_audio(audio_fp, sr=16000)
        audio_dur = librosa.get_duration(path=audio_fp)

        # Generate keypoint proposals
        keypoints_0 = np.arange(0, audio_dur)
        candidate_keypoints_1 = generate_candidate_keypoints(
            keypoints_0, args.max_error, audio_dur, args.n_delays_to_try
        )

        best_keypoints_1 = keypoints_0
        best_similarity = -np.infty

        # Search for the best keypoint proposal
        for keypoints_1 in candidate_keypoints_1:
            similarity = compute_similarity(audio, sr, keypoints_0, keypoints_1, args.window_size, model)
            if similarity > best_similarity:
                best_similarity = similarity
                best_keypoints_1 = keypoints_1

        # Save best keypoint proposal for the file
        fn = os.path.basename(audio_fp)
        keypoints_0 = list(keypoints_0)
        keypoints_1 = list(best_keypoints_1)
        predictions["Filename"].extend([fn for _ in keypoints_0])
        predictions["Time Channel 0"].extend(keypoints_0)
        predictions["Time Channel 1"].extend(keypoints_1)

    # Save all predictions
    predictions = pd.DataFrame(predictions)
    output_fp = os.path.join(args.output_dir, "predictions.csv")
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
        help="controls how many candidate keypoint sets to try before choosing the best. Inference run time is quadratic in this number",
    )
    parser.add_argument("--pretrained-fp", type=str, required=True, help="path to model weights if using pretraining")
    parser.add_argument(
        "--beats-fp",
        type=str,
        default="BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        help="Path to beats checkpoint, can be obtained from https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
