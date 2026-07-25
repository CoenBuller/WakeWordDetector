import torchaudio 
import os
import matplotlib.pyplot as plt

import soundfile as sf
import torch

from Augment.augment_config import AugmentConfig


def load_audio(path):
    data, sr = sf.read(path, dtype='float32')  # shape: (n_samples,) or (n_samples, channels)
    if data.ndim == 1:
        data = data[:, None]
    tensor = torch.from_numpy(data.T)  # torchaudio convention: (channels, samples)
    return tensor, sr


cfg = AugmentConfig()
sample_num = 10
mel_kwargs = {"hop_length": cfg.hop_length, 
              "n_fft": cfg.n_fft}

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

pos_folder = os.path.join("Training", "Data", "Positive")
neg_folder = os.path.join("Training", "Data", "Negative")

pos_sample = os.path.join(pos_folder, os.listdir(pos_folder)[sample_num])
neg_sample = os.path.join(neg_folder, os.listdir(neg_folder)[50])

pos_signal, sr = load_audio(pos_sample)
neg_signal, sr = load_audio(neg_sample)

pos_mel = torch.log(mel_transform(pos_signal)[0])
neg_mel = torch.log(mel_transform(neg_signal)[0])

pos_mel[pos_mel == -torch.inf] = torch.kthvalue(pos_mel.unique(), k=2)[0]
neg_mel[neg_mel == -torch.inf] = torch.kthvalue(neg_mel.unique(), k=2)[0]

pos_mel = (pos_mel - pos_mel.min()) / (pos_mel.max() - pos_mel.min() + 1e-8)
neg_mel = (neg_mel - neg_mel.min()) / (neg_mel.max() - neg_mel.min() + 1e-8)

fig, ax = plt.subplots(1, 2)
ax[0].set_title("Positive sample")
ax[0].imshow(pos_mel)

ax[1].set_title("Negative sample")
ax[1].imshow(neg_mel)
plt.show()
