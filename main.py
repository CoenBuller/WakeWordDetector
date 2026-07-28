import sounddevice as sd
import torch
import queue
import numpy as np
from InferenceConfig import InferenceConfig
from Augment.augment_config import AugmentConfig
from Augment.augment_pipeline import AugPipeline
from torchaudio.transforms import MelSpectrogram

def configure_sounddevice(cfg):
    sd.default.samplerate = cfg.sr
    sd.default.channels = 1  # Process mono for simplicity
    sd.default.blocksize = int(cfg.sr * cfg.callback_time)
    sd.default.dtype = cfg.dtype

def load_model(cfg):
    model = torch.load(f=cfg.model_path, map_location=cfg.device, weights_only=False)
    model.eval()
    return model

def process_one_frame(chunk, buffer, model, spec, cfg):
    # Ensure mono
    if chunk.ndim > 1:
        chunk = chunk.mean(axis=1, keepdims=True)
    
    # Flatten and add to buffer
    chunk_flat = chunk.flatten()
    buffer.extend(chunk_flat)
    
    # Keep only last cfg.sr samples
    if len(buffer) >= cfg.sr:
        buffer = buffer[-cfg.sr:]
    
        # Take exactly cfg.sr samples
        audio_segment = np.array(buffer[:cfg.sr], dtype=np.float32)
        tensor = torch.tensor(audio_segment, dtype=torch.float32)
        # Add batch dimension
        mel = spec(tensor.unsqueeze(0))
        
        with torch.no_grad():
            pred = model(mel)
        
        # Assuming pred is a tensor, get value
        pred_value = pred.item() if torch.is_tensor(pred) else pred
        
        if pred_value >= cfg.conf_thresh:
            print(pred_value)
            return True, buffer
    
    return False, buffer

def main():

    # Print host APIs available on your laptop (e.g., MME, WASAPI, DirectSound, ALSA, CoreAudio)
    print("--- Available Host APIs ---")
    for api in sd.query_hostapis():
        print(f"Index {api['name']}: {api['name']}")

    print("\n--- Detailed Device Scan ---")
    devices = sd.query_devices()
    for index, dev in enumerate(devices):
        # This reveals hidden or native channels that might not show up on your default system list
        print(f"ID {index}: {dev['name']} | Input Chans: {dev['max_input_channels']} | API: {sd.query_hostapis(dev['hostapi'])['name']}")

    d = input("\nType the index number of input device you want to use (e.g. 1).")
    cfg = InferenceConfig(input_device=int(d))
    spec_cfg = AugmentConfig()
    configure_sounddevice(cfg)
    model = load_model(cfg)
    audio_queue = queue.Queue(maxsize=10)
    buffer = []  # Use list for buffer
    spec = AugPipeline(
                    cfg=spec_cfg,
                    training=False
                    )
    
    spec = spec.to(cfg.device)
    model = model.to(cfg.device)

    def sd_callback(indata, frames, cb_time, status):
        if status:
            print(f"Status: {status}")
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # Skip frame if queue is full
            pass

    with sd.InputStream(device=cfg.input_device, callback=sd_callback):
        print("Listening... Press Ctrl+C to stop")
        try:
            triggered = True
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                
                trigger, buffer = process_one_frame(
                                            chunk=chunk,
                                            buffer=buffer,
                                            model=model,
                                            spec=spec,
                                            cfg=cfg
                                            )   
                                        
                if trigger:
                    print("How may I be of service my lord?")
                    buffer = []
                    triggered = True
                elif triggered == True:
                    triggered = False
                    print("Awaiting your command my lord.")
                    
        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == "__main__":
    main()