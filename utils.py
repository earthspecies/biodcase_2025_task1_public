import numpy as np
from torch import nn
import torchaudio

def remove_duplicate_keypoints(keypoints_0, keypoints_1):
    """
    When keypoints are generated, some timestamp values are repeated to avoid having timestamps beyond clip boundaries. This function removes duplicate keypoints at the beginning and end of the proposed keypoints.

    Args:
    keypoints_0 (np.ndarray): A 1D NumPy array of keypoints from the first set.
    keypoints_1 (np.ndarray): A 1D NumPy array of keypoints from the second set.

    Returns:
    tuple[np.ndarray, np.ndarray]: The cleaned keypoint arrays with duplicates at the boundaries removed.
    """    
    keypoints_min = min(np.amin(keypoints_0), np.amin(keypoints_1))
    keypoints_max = max(np.amax(keypoints_0), np.amax(keypoints_1))
        
    k0start = np.nonzero(keypoints_0 == keypoints_min)[0]
    k0start = np.amax(k0start) if len(k0start) else 0
    k1start = np.nonzero(keypoints_1 == keypoints_min)[0]
    k1start = np.amax(k1start) if len(k1start) else 0
    
    k0end = np.nonzero(keypoints_0 == keypoints_max)[0]
    k0end = np.amin(k0end) if len(k0end) else len(keypoints_0)+1
    k1end = np.nonzero(keypoints_1 == keypoints_max)[0]
    k1end = np.amin(k1end) if len(k1end) else len(keypoints_1)+1
    
    start_idx = max(k0start, k1start)
    end_idx = min(k0end,k1end)
    keypoints_0 = keypoints_0[start_idx:end_idx]
    keypoints_1 = keypoints_1[start_idx:end_idx]
    return keypoints_0, keypoints_1

def generate_candidate_keypoints(keypoints_0, max_allowed_error, duration, n_delays_to_try):
    """
    Generate candidate keypoint sets, assuming desynchronization consists of an offset + linear drift

    Args:
    keypoints_0 (np.ndarray): A 1D NumPy array containing keypoints from channel 0.
    max_allowed_error (float): The maximum allowable difference in seconds between keypoints_0 and generated keypoints.
    duration (float): The duration of the audio file in seconds.
    n_delays_to_try (int): The number of candidate keypoint sets to generate.

    Returns:
    list[np.ndarray]: A list of NumPy arrays, each representing a modified keypoint set with 
                      the same shape as keypoints_0 and within the allowable error.
    """
    
    candidate_offsets = np.linspace(-max_allowed_error, max_allowed_error, num=n_delays_to_try)
    max_slope = max_allowed_error / duration
    candidate_slopes = np.linspace(1-max_slope, 1+max_slope, num=n_delays_to_try)
    
    candidate_keypoints = []
    for offset in candidate_offsets:
        for slope in candidate_slopes:
            keypoints_1 = slope*keypoints_0 + offset
            keypoints_1 = np.maximum(keypoints_1, 0)
            keypoints_1 = np.minimum(keypoints_1, duration)
            observed_error = np.amax(np.abs(keypoints_1 - keypoints_0))
            if observed_error <= max_allowed_error:
                candidate_keypoints.append(keypoints_1)
    
    return candidate_keypoints

def load_audio(audio_fp, sr=None):
    """
    Load an audio file and optionally resample it to a specified sample rate.

    Args:
    audio_fp (str): File path to the audio file.
    sr (int, optional): Desired sample rate. If None, the original sample rate is used.

    Returns:
    tuple[torch.Tensor, int]: A tuple containing the loaded audio as a Torch tensor and the sample rate.
    """
    audio, orig_sr = torchaudio.load(audio_fp)
    if sr is None:
        return audio, orig_sr
    else:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
        return audio, sr

def pad_to_dur(audio, sr, dur_sec):
    """
    Pad or crop an audio tensor to match a specified duration.

    Args:
    audio (torch.Tensor): Input audio tensor with shape [..., N], where N is the number of samples.
    sr (int): Sample rate of the audio.
    dur_sec (float): Desired duration of the output audio in seconds.

    Returns:
    torch.Tensor: Audio tensor padded or cropped to have a duration of sr * dur_sec samples.
    """
    desired_dur_samples = int(sr*dur_sec)
    audio_dur_samples = audio.size(-1)
    pad = desired_dur_samples - audio_dur_samples
    if pad>0:
        audio = nn.functional.pad(audio, (0,pad))
    audio = audio[...,:desired_dur_samples]
    return audio