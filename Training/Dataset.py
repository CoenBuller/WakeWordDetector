import os
import pandas as pd
import torch
import soundfile as sf
import numpy as np

from torch.utils.data import Dataset
from Augment.augment_config import AugmentConfig
from torchaudio.transforms import MelSpectrogram

class TrainDataset(Dataset):
    def __init__(self, items, target_samplerate, p_silence=0.2, rng=np.random.default_rng(seed=49), transformer=None, target_transformer=None):

        self.items = items
        self.sr = target_samplerate
        self.p_silence = p_silence
        self.rng = rng
        self.transformer = transformer
        self.target_transformer = target_transformer

    def load_audio(self, path):
        data, _ = sf.read(path, dtype='float32') 
        tensor = torch.from_numpy(data)
        return tensor

    def __getitem__(self, idx):

        if self.rng.random(size=1) <= self.p_silence:
            audio = torch.zeros(self.sr, dtype=torch.float32)
            label = 0
        else:
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
    def __init__(self, items):
        self.items = items

        cfg = AugmentConfig()
        self.spec = MelSpectrogram(sample_rate=cfg.sr, 
                                   n_fft=cfg.n_fft, 
                                   hop_length=cfg.hop_length)

    def load_audio(self, path):
        data, _ = sf.read(path, dtype='float32') 
        tensor = torch.from_numpy(data)
        return tensor

    def __getitem__(self, idx):

        item = self.items.iloc[idx]
        path, label = item["path"], item["label"]
        audio = self.load_audio(path=path) # Audio has shape (n_samples) with the target samplerate
        mel = self.spec(audio)
    
        try:
            # Replace -inf values to the second lowest value in tensor
            replace_val = torch.kthvalue(mel.unique(), k=2)[0]
            mel = torch.where(mel == -torch.inf, replace_val, mel)
        except RuntimeError: 
            if -torch.inf in mel:
                mel = torch.zeros_like(mel)

            # Convert to [0, 1] range for consistent input
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            mel = torch.clamp(mel, min=0, max=1)         

        return mel, label

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

    
