from dataclasses import dataclass 

@dataclass
class AugmentConfig():
    # log-mel spectogram parameters
    n_fft: int = 1024
    sr: int = 15_872
    hop_length: int = 256    
    n_mels: int = 256

    # Waveform augment parameters 
    # Time shift
    p_time_shift: float = 0.4 # probability that a timeshift is applied
    time_shift: float = 0.1 # Keep it low as the wakeword must remain clearly present.

    # Pitch shift
    p_pitch_shift: float = 0.6 
    pitch_shift: int = 3
    
    # Volume scale 
    p_volume_scale: float = 0.5
    volume_scale: float = 0.5

    # Noise
    background_folder: str = "Training/Data/Background"
    p_noise: float = 0.7
    snr: tuple[int, int] = (5, 20)

    # RIR
    p_rir: float = 0.2 

    # Spectral augments
    spec_augment_prob:  float = 0.5
    n_freq_masks:       int   = 2
    freq_mask_param:    int   = 1
    n_time_masks:       int   = 2
    time_mask_param:    int   = 5

    # Random seed 
    seed: int = 49

    # Chance that the audio file is just silence
    p_silence: float = 0.2

    # Training Sequence
    batch_size: int = 64