import json
from pathlib import Path

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

AUDIO_PATH = "data/scenes/scene_001.wav"
OUTPUT_PATH = "data/context/scene_001_context.json"

SAMPLE_RATE = 16000

# Smaller overlapping windows give us better temporal resolution.
WINDOW_SECONDS = 5
HOP_SECONDS = 2.5

# We use a lower threshold because some relevant sounds,
# especially background sounds, can receive lower AST scores.
THRESHOLD = 0.05


# ============================================================
# AUDIOSET LABELS WE CARE ABOUT
# ============================================================

# Multiple AudioSet labels can represent the same concept.
# We map them into the labels used by our dataset.

LABEL_MAP = {

    # ---------------- DOG ----------------
    "Dog": "dog",
    "Bark": "dog",
    "Bow-wow": "dog",
    "Whimper (dog)": "dog",
    "Yip": "dog",
    "Growling": "dog",

    # ---------------- BIRDS ----------------
    "Bird": "chirping_birds",
    "Chirp, tweet": "chirping_birds",
    "Bird vocalization, bird call, bird song": "chirping_birds",

    # ---------------- WIND ----------------
    "Wind": "wind",
    "Wind noise (microphone)": "wind",

    # ---------------- FOOTSTEPS ----------------
    "Footsteps": "footsteps",
    "Walk, footsteps": "footsteps",

    # ---------------- WEATHER ----------------
    "Rain": "rain",
    "Raindrop": "water_drops",
    "Thunder": "thunderstorm",
    "Thunderstorm": "thunderstorm",

    # ---------------- VEHICLES ----------------
    "Vehicle": "engine",
    "Engine": "engine",
    "Car": "engine",
    "Vehicle horn, car horn, honking": "car_horn",
    "Siren": "siren",
    "Train": "train",
    "Helicopter": "helicopter",

    # ---------------- ANIMALS ----------------
    "Cow": "cow",
    "Cowbell": "cow",
    "Sheep": "sheep",
    "Pig": "pig",
    "Rooster": "rooster",
    "Chicken, rooster": "rooster",
    "Frog": "frog",
    "Cricket": "crickets",
    "Crow": "crow",
    "Insect": "insects",

    # ---------------- INDOOR ----------------
    "Keyboard typing": "keyboard_typing",
    "Typing": "keyboard_typing",
    "Mouse": "mouse_click",
    "Mouse click": "mouse_click",

    "Laughing": "laughing",
    "Cough": "coughing",
    "Sneeze": "sneezing",
    "Snoring": "snoring",

    # ---------------- OBJECTS ----------------
    "Glass": "glass_breaking",
    "Fireworks": "fireworks",

    # ---------------- HOUSEHOLD ----------------
    "Vacuum cleaner": "vacuum_cleaner",
    "Washing machine": "washing_machine",
    "Toilet flush": "toilet_flush",

    # ---------------- OTHER ----------------
    "Church bell": "church_bells",
    "Clapping": "clapping",
    "Door": "door_wood_knock",
    "Knock": "door_wood_knock",

    "Breathing": "breathing",
    "Drinking, sipping": "drinking_sipping",
    "Brushing teeth": "brushing_teeth",
    "Can opening": "can_opening",

    "Crackling fire": "crackling_fire",
    "Chainsaw": "chainsaw",
    "Hand saw": "hand_saw",
    "Pouring water": "pouring_water",
    "Sea waves": "sea_waves",
}


# ============================================================
# CREATE AUDIO WINDOWS
# ============================================================

def create_windows(audio, sample_rate):

    window_samples = int(
        WINDOW_SECONDS * sample_rate
    )

    hop_samples = int(
        HOP_SECONDS * sample_rate
    )

    windows = []

    start_sample = 0

    while start_sample < len(audio):

        end_sample = start_sample + window_samples

        chunk = audio[start_sample:end_sample]

        if len(chunk) == 0:
            break

        start_time = start_sample / sample_rate
        end_time = min(
            end_sample / sample_rate,
            len(audio) / sample_rate
        )

        windows.append({
            "audio": chunk,
            "start": start_time,
            "end": end_time
        })

        start_sample += hop_samples

    return windows


# ============================================================
# GET DATASET LABELS FROM AST OUTPUT
# ============================================================

def get_mapped_predictions(
    probabilities,
    model,
):

    predictions = []

    for index, score in enumerate(probabilities):

        score = score.item()

        ast_label = model.config.id2label[index]

        dataset_label = LABEL_MAP.get(ast_label)

        if dataset_label is None:
            continue

        if score < THRESHOLD:
            continue

        predictions.append({
            "label": dataset_label,
            "ast_label": ast_label,
            "confidence": score
        })

    return predictions


# ============================================================
# MERGE OVERLAPPING DETECTIONS
# ============================================================

def merge_detections(detections):

    grouped = {}

    for detection in detections:

        label = detection["label"]

        if label not in grouped:
            grouped[label] = []

        grouped[label].append(detection)

    merged = []

    for label, items in grouped.items():

        items.sort(
            key=lambda x: x["start"]
        )

        current = None

        for item in items:

            if current is None:

                current = {
                    "label": label,
                    "start": item["start"],
                    "end": item["end"],
                    "confidence": item["confidence"]
                }

                continue

            # If this detection overlaps or touches
            # the current detection, merge them.
            if item["start"] <= current["end"]:

                current["end"] = max(
                    current["end"],
                    item["end"]
                )

                current["confidence"] = max(
                    current["confidence"],
                    item["confidence"]
                )

            else:

                merged.append(current)

                current = {
                    "label": label,
                    "start": item["start"],
                    "end": item["end"],
                    "confidence": item["confidence"]
                }

        if current is not None:
            merged.append(current)

    merged.sort(
        key=lambda x: (
            x["start"],
            x["label"]
        )
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO CONTEXT LAYER - IMPROVED AST DETECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("\nLoading audio...")

    audio, sample_rate = librosa.load(
        AUDIO_PATH,
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = len(audio) / sample_rate

    print(f"Audio: {AUDIO_PATH}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {duration:.2f} seconds")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading AST model...")

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    print("AST loaded successfully.")

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    windows = create_windows(
        audio,
        sample_rate
    )

    print(
        f"\nCreated {len(windows)} "
        "overlapping windows."
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    raw_detections = []

    for window_number, window in enumerate(
        windows,
        start=1
    ):

        start = window["start"]
        end = window["end"]

        print("\n" + "-" * 70)

        print(
            f"WINDOW {window_number}: "
            f"{start:.2f}s -> {end:.2f}s"
        )

        print("-" * 70)

        inputs = feature_extractor(
            window["audio"],
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.sigmoid(
                outputs.logits
            )[0]

        mapped_predictions = (
            get_mapped_predictions(
                probabilities,
                model
            )
        )

        if not mapped_predictions:

            print(
                "No relevant dataset labels "
                f"above {THRESHOLD}"
            )

            continue

        mapped_predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        for prediction in mapped_predictions:

            print(
                f"{prediction['label']:<20} "
                f"{prediction['ast_label']:<40} "
                f"{prediction['confidence']:.4f}"
            )

            raw_detections.append({
                "label": prediction["label"],
                "start": start,
                "end": end,
                "confidence": prediction["confidence"]
            })

    # --------------------------------------------------------
    # Merge overlapping windows
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MERGING OVERLAPPING DETECTIONS")
    print("=" * 70)

    merged_events = merge_detections(
        raw_detections
    )

    # --------------------------------------------------------
    # Assign event IDs
    # --------------------------------------------------------

    events = []

    for index, event in enumerate(
        merged_events,
        start=1
    ):

        events.append({
            "event_id": f"pred_{index}",
            "label": event["label"],
            "start": round(event["start"], 2),
            "end": round(event["end"], 2),
            "confidence": round(
                event["confidence"],
                4
            )
        })

    # --------------------------------------------------------
    # Context JSON
    # --------------------------------------------------------

    context = {
        "scene_id": Path(AUDIO_PATH).stem,
        "duration": round(duration, 2),
        "source": "AST sliding window perception",
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "threshold": THRESHOLD,
        "events": events
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            context,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CONTEXT")
    print("=" * 70)

    print(
        f"Total predicted events: "
        f"{len(events)}"
    )

    if events:

        for event in events:

            print(
                f"{event['event_id']:<10}"
                f"{event['label']:<22}"
                f"{event['start']:>6.2f}s -> "
                f"{event['end']:>6.2f}s "
                f"(confidence="
                f"{event['confidence']:.3f})"
            )

    else:

        print("No events detected.")

    print("\nContext JSON saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("IMPROVED AST DETECTOR COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()