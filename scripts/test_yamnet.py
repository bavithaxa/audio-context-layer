import os
import csv

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AUDIO_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "scenes",
    "scene_001.wav"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET_SAMPLE_RATE = 16000
TOP_K = 10

# YAMNet uses approximately 0.48 second frame spacing
FRAME_HOP = 0.48
FRAME_DURATION = 0.96


# ---------------------------------------------------------
# Load YAMNet
# ---------------------------------------------------------

print("Loading YAMNet...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet loaded.")


# ---------------------------------------------------------
# Load audio
# ---------------------------------------------------------

print(f"\nLoading audio: {AUDIO_PATH}")

waveform, sample_rate = sf.read(AUDIO_PATH)

# Convert stereo to mono
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)

waveform = waveform.astype(np.float32)

duration = len(waveform) / sample_rate

print(f"Original sample rate: {sample_rate}")
print(f"Audio duration: {duration:.2f} seconds")


# ---------------------------------------------------------
# Resample to 16 kHz
# ---------------------------------------------------------

if sample_rate != TARGET_SAMPLE_RATE:

    print(
        f"Resampling "
        f"{sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz"
    )

    waveform = librosa.resample(
        waveform,
        orig_sr=sample_rate,
        target_sr=TARGET_SAMPLE_RATE
    )

    sample_rate = TARGET_SAMPLE_RATE

    print("Resampling complete.")


# ---------------------------------------------------------
# Run YAMNet
# ---------------------------------------------------------

print("\nRunning YAMNet...")

waveform_tensor = tf.convert_to_tensor(
    waveform,
    dtype=tf.float32
)

scores, embeddings, spectrogram = yamnet_model(
    waveform_tensor
)

scores = scores.numpy()

print(f"Number of YAMNet frames: {scores.shape[0]}")
print(f"Number of classes: {scores.shape[1]}")


# ---------------------------------------------------------
# Load YAMNet class names correctly
# ---------------------------------------------------------

class_map_path = (
    yamnet_model
    .class_map_path()
    .numpy()
    .decode("utf-8")
)

print(f"\nClass map: {class_map_path}")

class_names = []

with open(
    class_map_path,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        class_names.append(
            row["display_name"]
        )


print(f"Loaded {len(class_names)} class names.")


# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------

if len(class_names) != scores.shape[1]:

    raise ValueError(
        f"Class count mismatch: "
        f"{len(class_names)} names vs "
        f"{scores.shape[1]} model outputs"
    )


# ---------------------------------------------------------
# Frame level predictions
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("YAMNet frame level predictions")
print("=" * 75)

for frame_index, frame_scores in enumerate(scores):

    top_indices = np.argsort(
        frame_scores
    )[::-1][:TOP_K]

    start_time = frame_index * FRAME_HOP
    end_time = start_time + FRAME_DURATION

    print(
        f"\nFrame {frame_index + 1}: "
        f"{start_time:.2f}s - {end_time:.2f}s"
    )

    for rank, class_index in enumerate(
        top_indices,
        start=1
    ):

        score = frame_scores[class_index]

        print(
            f"{rank:2d}. "
            f"{class_names[class_index]:40s} "
            f"{score:.4f}"
        )


# ---------------------------------------------------------
# Overall predictions
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("Overall top YAMNet predictions")
print("=" * 75)

mean_scores = np.mean(
    scores,
    axis=0
)

top_indices = np.argsort(
    mean_scores
)[::-1][:20]

for rank, class_index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"{class_names[class_index]:40s} "
        f"{mean_scores[class_index]:.4f}"
    )


print("\nDone.")