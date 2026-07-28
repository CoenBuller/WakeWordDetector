import os
import pandas as pd
import torch
import soundfile as sf
import numpy as np

from torch.utils.data import Dataset
from scipy.signal import find_peaks

class TrainDataset(Dataset):
    def __init__(self, items, target_samplerate, rng=np.random.default_rng(seed=49), transformer=None, target_transformer=None):

        self.sr = target_samplerate
        self.rng = rng
        self.transformer = transformer
        self.target_transformer = target_transformer
        self.items = self._balance_classes(items=items, frac=0.4)

    def _find_valley_after_first_peak(self, audio: np.ndarray, sr: int, smooth_ms: float = 20.0):
        # 1. Rectify + smooth to get an amplitude envelope
        envelope = np.abs(audio)
        win_len = max(1, int(sr * smooth_ms / 1000))
        kernel = np.ones(win_len) / win_len
        envelope = np.convolve(envelope, kernel, mode="same")

        # 2. Find peaks, filtering out small wobbles inside a syllable
        min_gap = int(sr * 0.08)  # peaks closer than 80ms are treated as one
        peaks, props = find_peaks(
            envelope,
            distance=min_gap,
            prominence=0.15 * envelope.max(),
        )

        if len(peaks) < 2:
            return None, None, None
            raise ValueError(f"Found {len(peaks)} peak(s), expected 2 — check smooth_ms/prominence")

        first_peak, second_peak = peaks[0], peaks[1]

        # 3. Valley = envelope minimum strictly between the two peaks
        valley_idx = first_peak + np.argmin(envelope[first_peak:second_peak])

        return valley_idx, envelope, peaks

    def _balance_classes(self, items, frac: float = 0.4):
        pos_df = items[items["label"] == 1]
        neg_df = items[items["label"] == 0]
        
        len_pos = len(pos_df)
        len_neg = len(neg_df)

        print(f"Number of Positive samples: {len_pos} | Number of Negative samples: {len_neg}")
        
        if len_pos == 0:
            print("Warning: No positive samples. No balancing performed.")
            return items
        
        factor = int((frac / (1 - frac)) * len_neg / len_pos)
        
        pos_paths = pos_df["path"].tolist()
        oversampled_pos_paths = pos_paths * factor
        
        neg_paths = neg_df["path"].tolist()
        
        all_paths = oversampled_pos_paths + neg_paths        
        labels = [1] * len(oversampled_pos_paths) + [0] * len(neg_paths)
        balanced_df = pd.DataFrame({"path": all_paths, "label": labels})
        
        print(f"Balanced dataset: {len(oversampled_pos_paths)} positives, {len(neg_paths)} negatives")
        return balanced_df


    def load_audio(self, path):
        data, _ = sf.read(path, dtype='float32') 
        tensor = torch.from_numpy(data)
        return tensor

    def __getitem__(self, idx):


        item = self.items.iloc[idx]
        path, label = item["path"], item["label"]
        audio = self.load_audio(path=path) # Audio has shape (n_samples) with the target samplerate

        # Apply transformations if given. They must be callable transformations
        if self.transformer:
            audio = self.transformer(audio)
        if self.target_transformer:
            label = self.target_transformer(label)

        return audio, label
    
    def __len__(self):
        return len(self.items)

    def return_labels(self):
        return self.items["label"].to_numpy()

class ValidateDataset(Dataset):
    def __init__(self, items, transformer, augmentor=None, n_augments=50):
        self.transformer = transformer
        self.items = self._build(items=items, augmentor=augmentor, n_augments=n_augments)

    def load_audio(self, path):
        data, _ = sf.read(path, dtype='float32') 
        tensor = torch.from_numpy(data)
        return tensor

    def _build(self, items, augmentor, n_augments):
        specs = []
        labels = []

        for _, item in items.iterrows():
            audio = self.load_audio(path=item["path"])
            specs.append(self.transformer(audio))
            labels.append(item["label"])

        if augmentor is not None:
            pos_items = items[items["label"] == 1]
            for _, item in pos_items.iterrows():
                audio = self.load_audio(path=item["path"])
                for _ in range(n_augments):
                    specs.append(augmentor(audio))
                    labels.append(1)

            print(
                f"Validation set: added {n_augments * len(pos_items)} augmented positive "
                f"samples (from {len(pos_items)} originals)"
            )

        return pd.DataFrame({"spec": specs, "label": labels})

    def __getitem__(self, idx):
        item = self.items.iloc[idx]
        return item["spec"], item["label"]

    def __len__(self):
        return len(self.items)



class LoadDataset(Dataset):
    """Initial dataset structure used to create labels and will be used to create train test split"""
    def __init__(self, data_folder, rng=np.random.default_rng(seed=49)):

        self.dir = data_folder
        self.rng = rng
        self.labels = []
        self.paths = []
        self.items = self.__makelabels()


    def __makelabels(self):
        folders = [f for f in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, f))]
        for folder in folders:
            folder_path = os.path.join(self.dir, folder)

            # It will be for binary classification
            if folder == "Positive": 
                label = 1
            elif folder == "Negative":
                label = 0
            else: 
                continue

            for file in os.listdir(folder_path):
                path = os.path.join(folder_path, file)
                self.paths.append(path)
                self.labels.append(label) 
        
        return pd.DataFrame(data={"path": self.paths, "label": self.labels})
    
    def __getitem__(self, idx):

        item = self.items.iloc[idx]
        path, label = item["path"], item["label"]

        return path, label

    def __getitems__(self, indices):
        items = self.items.iloc[indices]
        return items
        
    def __len__(self):
        return len(self.items)

    def return_labels(self):
        return self.items["label"]