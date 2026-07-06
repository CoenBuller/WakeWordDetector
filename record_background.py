import sounddevice as sd
import os
from scipy.io import wavfile

sr = 44100 # hz
seconds = 60*10 # duration

print("Recording")
audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
print("Finished recording")

counter = 0
save_dir = "Training/Data/Background"
for i in range(seconds):
    path = os.path.join(save_dir, f"room_background_{counter}.wav")
    wavfile.write(path, sr, audio)
