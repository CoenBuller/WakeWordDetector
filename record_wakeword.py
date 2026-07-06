import sounddevice as sd
import numpy as np
import os
from scipy.io import wavfile

default_input_device_info = sd.query_devices(kind='input')

# Print de naam en het aantal kanalen
device_name = default_input_device_info['name']
max_channels = default_input_device_info['max_input_channels']

print(f"Standaard invoerapparaat: {device_name}")
print(f"Aantal beschikbare ingangskanalen: {max_channels}")

sr = 48000 # hz
n_samples = 30 # number of samples
save_dir = "Training/Data/Positive"

for i in range(n_samples):
    input(f"Recording sample {i+1}/{n_samples}. Press enter to record.")
    print("Recording...")
    audio = sd.rec(sr, samplerate=sr, channels=32, dtype='float32')
    sd.wait()
    
    mono_audio = np.mean(audio, axis=1).astype(np.int16)
    path = os.path.join(save_dir, f"Jarvis_{i}.wav")
    wavfile.write(path, sr, audio)
    print(f"Saved sample {i+1}/{n_samples} to: {path}")

