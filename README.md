# artem-stuti
# Lightweight Offline Speech-to-Text for Medical Prescriptions (Edge AI)

## Overview

This project implements a **minimal-footprint, fully offline Speech-to-Text (STT) prototype** that converts **doctor dictation into medical prescription text**, optimized for **low-resource / edge devices**.

The focus is on:
- Engineering judgment
- Model footprint awareness
- Offline inference
- Medical vocabulary biasing

The system uses a **lightweight Whisper Tiny model**, runs entirely on CPU, and demonstrates **medical-domain adaptation via prompt biasing and post-processing**.

---

## Features

- ✅ Fully **offline** speech-to-text
- 🎙️ Input from **microphone** or **WAV file**
- 🏥 Medical prescription vocabulary handling
- ⚡ Fast cold start & low memory usage
- 📦 Small model footprint suitable for edge deployment

---

## Example

**Input (voice):**
> “Paracetamol 500 MG, twice a day, after food.”

**Output (text):**
> “Paracetamol 500 mg, twice a day, after food.”


---

## Chosen STT Engine

### Whisper Tiny (OpenAI)

**Why Whisper Tiny?**
- Smallest Whisper variant (~39 MB)
- Strong baseline accuracy
- Runs fully offline on CPU
- Simple Python integration
- Well-suited for rapid prototyping on edge devices

Alternative engines (e.g., Vosk) were considered, but Whisper Tiny provides **better accuracy with acceptable footprint** for this prototype.

---

## Medical Vocabulary Handling

Medical context is handled using **multiple lightweight techniques**:

1. **Prompt Biasing**
   - An `initial_prompt` biases decoding toward prescription-related terms:
     ```
     paracetamol, amoxicillin, milligram, tablet,
     once daily, twice daily, after food
     ```

2. **Post-processing Corrections**
   - Common misrecognitions corrected via string normalization
   - Spoken numbers mapped to numeric values
   - Example:
     - `"five hundred milligrams"` → `"500 mg"`
     - `"para c tamol"` → `"paracetamol"`

3. **Domain-Specific Vocabulary Map**
   - Ensures reliable recognition of at least **10 common medical terms**

This avoids retraining or large language models, keeping the system lightweight.

---

## Footprint Metrics (Prototype-Level)

| Metric            | Value (Approx.)         |
|-------------------|--------------------------|
| Model Size        | ~39 MB                   |
| RAM Usage         | ~120–150 MB              |
| Inference Mode    | Fully Offline (CPU)      |
| Cold Start Time   | ~2–3 seconds             |

> Metrics measured on a typical laptop CPU. Edge devices may vary.

---

## Installation

### Requirements

- Python 3.8+
- Microphone (for live input)

### Install Dependencies

```bash
pip install openai-whisper sounddevice scipy numpy
