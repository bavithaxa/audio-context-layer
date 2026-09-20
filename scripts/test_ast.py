import torch
import librosa

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

AUDIO_PATH = "data/scenes/scene_001.wav"


print("=" * 60)
print("AUDIO CONTEXT LAYER - AST TEST")
print("=" * 60)


# ------------------------------------------------------------
# Load audio
# ------------------------------------------------------------

print("\nLoading audio...")

audio, sample_rate = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True
)

print(f"Audio loaded successfully.")
print(f"Sample rate: {sample_rate}")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")


# ------------------------------------------------------------
# Load feature extractor
# ------------------------------------------------------------

print("\nLoading AST feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading AST model...")

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME
)

model.eval()


# ------------------------------------------------------------
# Prepare audio
# ------------------------------------------------------------

inputs = feature_extractor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
)


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

print("\nRunning inference...")

with torch.no_grad():

    outputs = model(
        **inputs
    )

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]


# ------------------------------------------------------------
# Get top predictions
# ------------------------------------------------------------

top_k = 15

top_values, top_indices = torch.topk(
    probabilities,
    k=top_k
)


print("\n--- TOP AUDIO PREDICTIONS ---")

for score, index in zip(
    top_values,
    top_indices
):

    label = model.config.id2label[
        index.item()
    ]

    print(
        f"{label:<35} "
        f"{score.item():.4f}"
    )


print("\n" + "=" * 60)
print("AST TEST COMPLETE")
print("=" * 60)