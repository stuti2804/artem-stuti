"""
Lightweight Offline Speech-to-Text for Medical Prescriptions
Using Whisper Tiny (Offline)

Run (MIC input):
    python stt_prescription.py

Run (WAV input):
    python stt_prescription.py --audio sample.wav
"""

import argparse
import re
import whisper
import time
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# -----------------------------
# Load Model ONCE (Cold Start Optimization)
# -----------------------------
print("⏳ Loading Whisper Tiny model...")
MODEL_LOAD_START = time.time()

model = whisper.load_model(
    "tiny",
    device="cpu",
    download_root="./models"
)

print(f"✅ Model loaded in {time.time() - MODEL_LOAD_START:.2f} seconds")

# -----------------------------
# Medical Vocabulary & Biasing
# -----------------------------
MEDICAL_TERMS = {
    "paracetamol": ["para c tamol", "paracitamol", "paracetemol"],
    "amoxicillin": ["amoxycillin", "amoxacillin"],
    "milligram": ["milli gram", "milligrams"],
    "tablet": ["tab let", "tablets"],
    "once daily": ["once daili", "one daily"],
    "twice daily": ["twice daili", "two times daily"],
    "after food": ["after meal", "after eating"]
}

NUMBER_MAP = {
    "one": "1",
    "two": "2",
    "three": "3",
    "five hundred": "500",
    "five hundred milligrams": "500 mg",
    "milligrams": "mg"
}

# -----------------------------
# Post-processing corrections
# -----------------------------
def medical_postprocess(text: str) -> str:
    text = text.lower()

    for k, v in NUMBER_MAP.items():
        text = text.replace(k, v)

    for correct, variants in MEDICAL_TERMS.items():
        for v in variants:
            text = text.replace(v, correct)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Microphone Recording
# -----------------------------
def record_audio(filename="recorded.wav", duration=10, fs=16000):
    print(f"\n🎙 Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved as", filename)
    return filename

# -----------------------------
# STT Logic (Optimized Decoding)
# -----------------------------
def transcribe(audio_path: str):
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        temperature=0,
        beam_size=1,
        condition_on_previous_text=False,
        initial_prompt=(
            "Medical prescription dictation. "
            "Common words: paracetamol, amoxicillin, "
            "milligram, tablet, once daily, twice daily, after food."
        )
    )

    raw_text = result["text"]
    final_text = medical_postprocess(raw_text)
    latency = time.time() - start_time

    return raw_text, final_text, latency

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to WAV audio file (optional)"
    )
    args = parser.parse_args()

    if args.audio:
        audio_file = args.audio
    else:
        audio_file = record_audio()

    raw, corrected, time_taken = transcribe(audio_file)

    print("\n--- RAW TRANSCRIPTION ---")
    print(raw)

    print("\n--- MEDICAL-CORRECTED OUTPUT ---")
    print(corrected)

    print("\n--- METRICS ---")
    print(f"Inference time: {time_taken:.2f} seconds")
    print("Model: Whisper Tiny (~39 MB)")
    print("Estimated RAM: ~120–150 MB")
