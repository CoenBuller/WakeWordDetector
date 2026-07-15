import torch
import soundfile as sf
import torchaudio.functional as F
import numpy as np
import librosa as lb
import os

from Augment.augment_config import AugmentConfig
from numpy.typing import NDArray
from torch import Tensor
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import add_noise


class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        cfg: AugmentConfig, 
        training=False,
        rng = np.random.default_rng(seed=49)
    ):
        super().__init__()

        self.spec = MelSpectrogram(sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

        self.cfg = cfg
        self.rng = rng

        if training:
            self.rir_audio = self._load_rir_files("Training/Data/RIR.wav", "Training/Data/RIR_2.wav")
            self.background_files = os.listdir(cfg.background_folder)


    # The "rir" methods are used to make the background noice, sound more like actual background noise. 
    def _load_rir_files(self, *paths):
        rirs = []
        for path in paths:
            rir, sr = sf.read(path)
            rir_tensor = torch.tensor(rir, dtype=torch.float32)
            rirs.append(rir_tensor)

        return rirs


    def forward(self, waveform: NDArray) -> torch.Tensor:

        if self.training:
            waveform_tensor = self._waveform_augment(waveform=waveform)
        else:
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

        # Convert to mel spectrogram
        spec = self.spec(waveform_tensor)

        # Apply SpecAugment
        spec = self._spec_augment(spec)

        return spec
    
    def _apply_rir_augment(self, audio: Tensor):
        choice = self.rng.integers(0, 2)
        rir = self.rir_audio[choice]
        augmented = F.fftconvolve(audio, rir)
        return augmented
    
    def _pitch_shift(self, audio: Tensor, n_steps: int) -> Tensor:
        steps = int(self.rng.integers(low=-n_steps, high=n_steps))
        # torchaudio.functional.pitch_shift takes (waveform, sample_rate, n_steps)
        return F.pitch_shift(
            waveform=audio,
            sample_rate=self.cfg.sr,
            n_steps=steps
        )
 
    # Volume scaling by multiplying the audio signal by a random gain factor within a specified range.
    def _volume_scale(self, audio: Tensor, gain: float) -> Tensor:
        gain = self.rng.uniform(1-gain, 1+gain)
        return torch.clamp(audio * gain, -1.0, 1.0)
    
    # Time shifting by rolling the audio array by a random number of samples.
    def _time_shift(self, audio: Tensor, shift: float) -> Tensor:
        # Shift is a fraction of total length, e.g. (-0.2, 0.2)
        shift = self.rng.uniform(-shift, shift)
        n = int(shift * len(audio))
        return torch.roll(audio, shifts=n, dims=0) 
    
    def _add_noise(self, audio: Tensor, snr: int):
        snr_tensor = torch.tensor(snr, dtype=torch.float32)
        choice = self.rng.integers(0, 2)

        # Add white noise
        if choice == 0:
            noise = self.rng.normal(loc=0, scale=1, size=len(audio)).astype(dtype=np.float32)
            noise = torch.from_numpy(noise)

        # Add a background file
        else:
            path = self.rng.choice(self.background_files)
            background, sr = sf.read(path)
            background = torch.tensor(background, dtype=torch.float32)
            noise = self._apply_rir_augment(audio=background)

        noisy_audio = add_noise(waveform=audio, noise=noise, snr=snr_tensor)
        return noisy_audio
    
    def _waveform_augment(self, waveform: NDArray) -> Tensor:
        # Move to Tensor immediately at the start of the augmentation pipeline
        tensor = torch.tensor(waveform, dtype=torch.float32)

        # Timeshift
        if self.rng.random(1) <= self.cfg.p_time_shift:
            tensor = self._time_shift(audio=tensor, shift=self.cfg.time_shift)
        
        # Pitch shift
        if self.rng.random(1) <= self.cfg.p_pitch_shift:
            tensor = self._pitch_shift(audio=tensor, n_steps=self.cfg.pitch_shift)
        
        # Volume scaling
        if self.rng.random(1) <= self.cfg.p_volume_scale:
            tensor = self._volume_scale(audio=tensor, gain=self.cfg.volume_scale)

        # RIR transformation
        if self.rng.random(1) <= self.cfg.p_rir:
            tensor = self._apply_rir_augment(audio=tensor)
        
        # Add background noise
        if self.rng.random(1) <= self.cfg.p_noise:
            snr = int(self.rng.uniform(self.cfg.snr[0], self.cfg.snr[1]))
            tensor = self._add_noise(audio=tensor, snr=snr)

        return tensor
    
    def _spec_augment(self, result: Tensor) -> Tensor:
        n_mels, n_frames = result.shape
        cfg = self.config
        fill = result.mean()
        for _ in range(self.rng.integers(low=0, high=self.cfg.n_freq_masks)):
            f  = self.rng.integers(0, n_mels-1)
            result[f, :] = fill          # freq masking is fine for claps

        for _ in range(self.rng.integers(low=0, high=self.cfg.n_time_masks)):
            t  = self.rng.integers(0, n_frames-1)
            result[:, t] = fill

        return result