#/// script
# requires-python = ">=3.14"
# dependencies = [
#     "coqui-tts>=0.27.5",
#     "torch",
#     "torchaudio",
# ]
# ///
import torch
from TTS.api import TTS

def generate_johnny_audio(text_to_speak: str, reference_audio: str, output_filename: str):
    print("Loading XTTS v2 model (this may download model files on first run)...")
    
    # Initialize the XTTS v2 model and load it to CUDA (GPU)
    # Note: XTTS v2 requires a GPU for reasonable speed.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"Cloning voice from '{reference_audio}' and generating speech...")
    
    # Generate speech by cloning from your reference wav/mp3 file
    tts.tts_to_file(
        text=text_to_speak,
        speaker_wav=reference_audio,
        language="en",
        file_path=output_filename
    )
    
    print(f"Success! Audio saved to {output_filename}")

if __name__ == "__main__":
    # Make sure 'johnny_sample.mp3' is in the same directory (or provide the full path)
    generate_johnny_audio(
        text_to_speak="Hey, V. Don't burn the lunch, choom.",
        reference_audio="johnny_sample.mp3",
        output_filename="johnny_output.mp3"
    )