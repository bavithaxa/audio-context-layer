import os
import numpy as np
import librosa
from panns_inference import AudioTagging, labels


AUDIO_PATH = "data/scenes/scene_001.wav"

CHECKPOINT_PATH = r"C:\Users\ASUS\panns_data\Cnn14_mAP=0.431.pth"

SAMPLE_RATE = 32000
WINDOW_SECONDS = 5.0


def main():

    print("=" * 60)
    print("PANNs CNN14 5 second window test")
    print("=" * 60)

    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(AUDIO_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    print("\nLoading audio...")

    audio, sr = librosa.load(
        AUDIO_PATH,
        sr=SAMPLE_RATE,
        mono=True
    )

    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio) / sr:.2f} seconds")

    # ---------------------------------------------------------
    # Take first 5 second window
    # ---------------------------------------------------------

    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)

    window = audio[:window_samples]

    if len(window) < window_samples:
        window = np.pad(
            window,
            (0, window_samples - len(window))
        )

    window = window.astype(np.float32)

    # Add batch dimension
    window = window[None, :]

    print("\nTesting window:")
    print("Start: 0.00 sec")
    print("End:   5.00 sec")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("\nLoading PANNs CNN14...")

    model = AudioTagging(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    )

    print("Model loaded.")

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    print("\nRunning inference...")

    clipwise_output, embedding = model.inference(window)

    scores = clipwise_output[0]

    print(f"\nNumber of classes: {len(scores)}")
    print(f"Embedding shape: {embedding.shape}")

    # ---------------------------------------------------------
    # Top predictions
    # ---------------------------------------------------------

    top_indices = np.argsort(scores)[::-1][:20]

    print("\nTop 20 predictions:")
    print("-" * 60)

    for rank, index in enumerate(top_indices, start=1):

        label = labels[index]
        score = float(scores[index])

        print(
            f"{rank:2d}. "
            f"{label:<45} "
            f"{score:.4f}"
        )

    print("\n" + "=" * 60)
    print("5 second PANNs test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()