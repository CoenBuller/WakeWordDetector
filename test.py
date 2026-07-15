import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.functional as F
import torch

from scipy.io import wavfile


path_rir = "Training/Data/RIR.wav"
path_speech = "Training/Data/Negative/non_wakeword_13.wav"
rir, sr = sf.read(path_rir)


speech, sr = sf.read(path_speech)
speech_tensor, rir_tensor = torch.tensor(speech, dtype=torch.float32), torch.tensor(rir, dtype=torch.float32)
augmented = F.fftconvolve(speech_tensor, rir_tensor).numpy()
wavfile.write("rir_augmented.wav", sr, augmented)
