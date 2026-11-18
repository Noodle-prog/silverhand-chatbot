# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "tts>=0.22.0",
# ]
# ///
from TTS.api import TTS

# Load a multi-speaker / voice-cloning model locally
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Generate speech by cloning from a reference wav file of Johnny
tts.tts_to_file(
    text="Wake the fuck up, V. We've got a city to burn.",
    speaker_wav="johnny_sample.mp3",
    language="en",
    file_path="johnny_output.mp3"
)