import torch
import soundfile as sf
import torchaudio.functional as F
import numpy as np
import os

from Augment.augment_config import AugmentConfig
from torch import Tensor
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import add_noise


class AugPipeline(torch.nn.Module):
    def __init__(
        self,
        cfg: AugmentConfig, 
        training=True,
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


    @torch.no_grad()
    def forward(self, waveform: Tensor) -> torch.Tensor:

        if self.training:
            waveform_tensor = self._waveform_augment(waveform=waveform)
        else:
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
    

        # Convert to log mel spectrum
        spec = torch.log(self.spec(waveform_tensor)) 

        # If the tensor is pure silence, the "try" block will fail, and we will return a tensor of zeros
        try:
            # Replace -inf values to the second lowest value in tensor
            replace_val = torch.kthvalue(spec.unique(), k=2)[0]
            spec = torch.where(spec == -torch.inf, replace_val, spec)
        except RuntimeError: 
            if -torch.inf in spec:
                spec = torch.zeros_like(spec)

            # Convert to [0, 1] range for consistent input
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            spec = torch.clamp(spec, min=0, max=1)  

        # Apply SpecAugment
        if self.training:
            spec = self._spec_augment(spec)

        return spec
    
    def _apply_rir_augment(self, audio: Tensor):
        choice = self.rng.integers(0, 2)
        rir = self.rir_audio[choice]
        augmented = F.fftconvolve(audio, rir)
        
        augmented_length = len(augmented)
        if augmented_length < self.cfg.sr:
            zeros = torch.zeros_like(augmented, dtype=torch.float32)
            zeros[:len(augmented)] = augmented
            augmented = zeros
            return augmented
        
        if augmented_length > self.cfg.sr:
            return augmented[-self.cfg.sr:]
        
        return augmented
 
    
    # Time shifting by rolling the audio array by a random number of samples.
    def _time_shift(self, audio: Tensor, shift: float) -> Tensor:
        # Shift is a fraction of total length, e.g. (-0.2, 0.2)
        shift = self.rng.uniform(-shift*2, shift)
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
            path = os.path.join(self.cfg.background_folder, path)
            background, sr = sf.read(path)
                
            background_tensor = torch.tensor(background, dtype=torch.float32)

            noise = self._apply_rir_augment(audio=background_tensor)

        noisy_audio = add_noise(waveform=audio, noise=noise, snr=snr_tensor)
        return noisy_audio
    
    def _waveform_augment(self, waveform: Tensor) -> Tensor:
        # Move to Tensor immediately at the start of the augmentation pipeline
        tensor = waveform

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
        fill = 0
        for _ in range(self.rng.integers(low=0, high=self.cfg.n_freq_masks)):
            f  = self.rng.integers(0, n_mels-1)
            result[f, :] = fill          # freq masking is fine for claps

        for _ in range(self.rng.integers(low=0, high=self.cfg.n_time_masks)):
            t  = self.rng.integers(0, n_frames-1)
            result[:, t] = fill

        return result