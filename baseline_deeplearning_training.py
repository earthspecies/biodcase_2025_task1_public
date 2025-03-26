"""
Train baseline deep learning system
Usage:
python baseline_deeplearning_training.py --output-dur=/path/to/desired/output/dir --train-dir=/path/to/train/dir --val-dir=/path/to/val/dir
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from models import BEATsEncoderAndMLP
from torch import nn
from tqdm import tqdm
from utils import load_audio, pad_to_dur, remove_duplicate_keypoints

device = "cuda" if torch.cuda.is_available() else "cpu"


class AudioPairDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading and processing paired audio segments from a dataset.

    Attributes:
    clip_dur (int): Duration of the extracted audio clips in seconds.
    sr (int): Sample rate of the audio.
    audio (list): A list of tuples containing paired audio segments (audio_0, audio_1).
    rng (np.random.Generator): A random number generator for dataset operations.
    """

    def __init__(self, data_dir):
        """
        Initialize the AudioPairDataset by loading audio files and their corresponding annotations.

        Args:
        data_dir (str): Path to the directory containing the 'annotations.csv' file and an 'audio' subdirectory.
        """
        # Read annotations
        annotations_fp = os.path.join(data_dir, "annotations.csv")
        annotations = pd.read_csv(annotations_fp)
        print(f"Loading Audio from {data_dir}")

        self.clip_dur = 1
        self.sr = 16000
        self.audio = []

        audio_fns = sorted(annotations["Filename"].unique())

        for audio_fn in tqdm(audio_fns):
            # Load audio
            audio_fp = os.path.join(data_dir, "audio", audio_fn)
            audio, sr = load_audio(audio_fp, sr=self.sr)

            # Load keypoints
            annotations_sub = annotations[annotations["Filename"] == audio_fn]
            keypoints_0 = annotations_sub["Time Channel 0"].values
            keypoints_1 = annotations_sub["Time Channel 1"].values

            # Remove duplicate keypoints at audio recording boundaries
            keypoints_0, keypoints_1 = remove_duplicate_keypoints(keypoints_0, keypoints_1)

            # Create examples
            for ii in range(len(keypoints_0) - 1):
                start_0 = int(sr * keypoints_0[ii])
                start_1 = int(sr * keypoints_1[ii])
                audio_0 = audio[
                    0, start_0 : start_0 + int(2 * self.clip_dur * sr)
                ]  # 2x duration; windowing is applied during training
                audio_1 = audio[1, start_1 : start_1 + int(2 * self.clip_dur * sr)]
                audio_0 = pad_to_dur(audio_0, sr, 2 * self.clip_dur)
                audio_1 = pad_to_dur(audio_1, sr, 2 * self.clip_dur)
                self.audio.append((audio_0, audio_1))

        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        flip = self.rng.binomial(1, 0.5)
        start = self.rng.integers(0, int(self.sr * self.clip_dur) - 1)
        end = start + int(self.sr * self.clip_dur)
        if flip:
            return self.audio[idx][1][start:end], self.audio[idx][0][start:end]
        else:
            return self.audio[idx][0][start:end], self.audio[idx][1][start:end]


def get_loss(model, audio_0, audio_1, pos_loss_weight=1):
    """
    Compute the binary cross-entropy loss for a model applied to audio pairs.

    Parameters:
    model (torch.Module): The neural network model used for embedding and computing the similarity.
    audio_0 (torch.Tensor): A tensor of shape (batch_size, samples) representing the first set of audio inputs.
    audio_1 (torch.Tensor): A tensor of shape (batch_size, samples) representing the second set of audio inputs.
    pos_loss_weight (float, optional): Weight for the positive loss term. Default is 1.

    Returns:
    torch.Tensor: The computed loss value.
    """
    batchsize = audio_0.size(0)

    e0 = model.embed(audio_0).unsqueeze(0)  # [1,B,N]
    e1 = model.embed(audio_1).unsqueeze(1)  # [B,1,N]

    e0 = torch.tile(e0, (batchsize, 1, 1))
    e1 = torch.tile(e1, (1, batchsize, 1))

    ee = torch.cat([e0, e1], dim=-1)  # [B,B,2N]
    logits = model.head(ee).squeeze(-1)  # [B,B]

    targets = torch.eye(batchsize, device=e0.device)  # [B,B]
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean", pos_weight=torch.tensor(pos_loss_weight * (batchsize - 1))
    )  # Reweight negative examples so that negative contribution == positive contribution regardless of batch size

    return loss


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize model
    model = BEATsEncoderAndMLP(args.beats_fp)
    model.to(device)

    # Initialize data
    train_dataset = AudioPairDataset(args.train_dir)
    val_dataset = AudioPairDataset(args.val_dir)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    for epoch in range(args.n_epochs):
        # Train steps
        print(f"Train Epoch {epoch}")
        model.train()
        model.freeze_encoder()
        losses = []
        for audio_0, audio_1 in tqdm(train_dataloader):
            loss = get_loss(model, audio_0, audio_1, pos_loss_weight=args.pos_loss_weight)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Train Loss: {np.mean(losses)}")

        # Val steps
        print(f"Val Epoch {epoch}")
        model.eval()
        losses = []
        for audio_0, audio_1 in tqdm(val_dataloader):
            with torch.no_grad():
                loss = get_loss(model, audio_0, audio_1, pos_loss_weight=args.pos_loss_weight)
                losses.append(loss.item())
        print(f"Val Loss: {np.mean(losses)}")

    print("Training complete")

    # Save weights
    output_fp = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), output_fp)
    print(f"Saved model to {output_fp}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="path to directory to put output files")
    parser.add_argument("--train-dir", type=str, required=True, help="path to train directory")
    parser.add_argument("--val-dir", type=str, required=True, help="path to val directory")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-loss-weight", type=float, default=1.0, help="Weight for positive component of loss")
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
