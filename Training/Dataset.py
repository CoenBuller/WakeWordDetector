import torchaudio 
import os
import pandas as pd
import torch
import soundfile as sf
import numpy as np

class WakeWordDataset():
    def __init__(self, data_folder, target_samplerate, p_silence=0.2, rng=np.random.default_rng(seed=49), transformer=None, target_transformer=None):
        self.target_samplerate = target_samplerate
        self.dir = data_folder
        self.transformer = transformer
        self.target_transformer = target_transformer
        self.labels = self.__makelabels()
        self.rng = rng
        self.p_silence = 0.2

    def load_audio(self, path):
        data, sr = sf.read(path, dtype='float32')           # shape: (n_samples,) or (n_samples, channels)
        if data.ndim != 1:
            data = np.mean(data, axis=1, keepdims=False)    # Convert to mono channels audio (n_samples,)

        resampled_data = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_samplerate)(data) # Resample to target samplerate to keep it consistent 
        tensor = torch.from_numpy(resampled_data)    
        return tensor

    def __makelabels(self):
        folders = [f for f in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, f))]
        paths, labels = [], []
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
                paths.append(path)
                labels.append(label) 
        
        return pd.DataFrame(data={"path": paths, "label": labels})
    
    def __getitem__(self, idx):
        
        if self.rng.random(size=1) <= self.p_silence:
            audio = torch.zeros(self.target_samplerate, dtype=torch.float32)
            label = 0
        else:
            item = self.labels.iloc[idx]
            path, label = item["path"], item["label"]
            audio = self.load_audio(path=path) # Audio has shape (n_samples) with the target samplerate

        # Apply transformations if given. They must be callable transformations
        if self.transformer:
            audio = self.transformer(audio)
        if self.target_transformer:
            label = self.target_transformer(label)

        return audio, label
        

    def __len__(self):
        return len(self.labels)

