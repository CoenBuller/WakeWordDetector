import numpy as np

from dataclasses import dataclass 

@dataclass
class InferenceConfig():
    # log-mel spectogram parameters
    n_fft: int = 1024
    hop_length: int = 256    
    n_mels: int = 256

    
    # Sounddevice parameters
    sr: int = 16_128
    channels: int = 1
    callback_time: float = 0.25 # s
    dtype: object = np.float32
    input_device: int = 1

    # Model
    model_path: str = "Jarvis.pt"
    device: str = "cpu"

    # Inference
    conf_thresh: float = 0.5 # between [0, 1]