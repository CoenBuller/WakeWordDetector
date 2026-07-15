import os
import soundfile as sf
import numpy as np
from tqdm import tqdm 


FOLDER_PATH = "C:/Users/coenb/speech-noise-dataset/noise_only"
assert os.path.isdir(FOLDER_PATH), f"Folder with original audio cannot be found at {FOLDER_PATH}"

SAVE_DIR = "Training/Data/Background"
os.makedirs(SAVE_DIR, exist_ok=True)  # Automatically creates the directory if it doesn't exist

audio_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(('.mp3', '.wav'))]
cont = input(f"Length of audio files is: {len(audio_files)}, do you want to continue (y/n)")
if cont == "n":
    exit()
# audio_files = np.random.choice(audio_files, size=1000)
print(f"Loaded in {len(audio_files)} audio file paths.")

# Keep global fragment counter outside so files don't overwrite each other
global_fragment_counter = 0
first_run_check = True

for audio_segment in tqdm(audio_files):
    path = os.path.join(FOLDER_PATH, audio_segment)
        
    try:
        audio, sr = sf.read(path) 
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
    except Exception as e:
        print(f"Skipping file {audio_segment}: {e}")
        continue
        
    duration_seconds = int(len(audio) / sr)
    
    for i in range(duration_seconds): 
        low = i * sr 
        high = (i + 1) * sr
        
        # Slice the 1-second fragment
        fragment = audio[low:high]
        
        # Ensure it's exactly 1 second long (ignores trailing tiny fragments)
        if len(fragment) < sr:
            continue
            
        p = os.path.join(SAVE_DIR, f"background_{global_fragment_counter}.wav")
        
        if first_run_check:
            print(f"\nPath that the first audio fragment will be saved to is: {p}")
            inp = input("Do you want to continue? (y/n): ").strip().lower()
            if inp == 'n':
                print("Aborted by user.")
                exit()
            first_run_check = False # Only ask once at the very start
            
        # Save cleanly using soundfile (librosa's stable backend companion)
        sf.write(p, fragment, sr)
        global_fragment_counter += 1  

print(f"\nSuccessfully finished! Created {global_fragment_counter} total 1-second fragments.")