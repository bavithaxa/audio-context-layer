import json
import os
import csv

import librosa
import numpy as np
import torch

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)

from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

WINDOW_INDEX_PATH = "data/processed/window_index.json"

OUTPUT_METRICS = "results/ast_metrics.json"
OUTPUT_PER_CLASS = "results/ast_per_class.csv"
OUTPUT_PREDICTIONS = "results/ast_window_predictions.json"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0

# Fixed baseline threshold.
# We can tune this on validation data later if needed.
THRESHOLD = 0.30


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_CLASSES = [
    "car_horn",
    "chirping_birds",
    "clock_tick",
    "cow",
    "crow",
    "dog",
    "door_wood_knock",
    "engine",
    "footsteps",
    "hen",
    "keyboard_typing",
    "laughing",
    "mouse_click",
    "rain",
    "rooster",
    "sheep",
    "siren",
    "thunderstorm",
    "water_drops",
    "wind"
]


# ============================================================
# AST / AUDIOSET LABEL MAPPING
# ============================================================

LABEL_MAPPING = {

    "car_horn": [
        "Honk",
        "Car horn, honking",
        "Car horn"
    ],

    "chirping_birds": [
        "Chirp, tweet",
        "Bird",
        "Bird vocalization, bird call, bird song"
    ],

    "clock_tick": [
        "Tick",
        "Tick-tock",
        "Clock"
    ],

    "cow": [
        "Moo",
        "Cow"
    ],

    "crow": [
        "Crow",
        "Caw"
    ],

    "dog": [
        "Dog",
        "Bark",
        "Bow-wow",
        "Whimper (dog)",
        "Yip"
    ],

    "door_wood_knock": [
        "Knock",
        "Door",
        "Wood"
    ],

    "engine": [
        "Engine",
        "Idling",
        "Vehicle"
    ],

    "footsteps": [
        "Walk, footsteps",
        "Footstep"
    ],

    "hen": [
        "Chicken, rooster",
        "Chicken",
        "Cluck"
    ],

    "keyboard_typing": [
        "Typing",
        "Computer keyboard"
    ],

    "laughing": [
        "Laughter"
    ],

    "mouse_click": [
        "Mouse click",
        "Click"
    ],

    "rain": [
        "Rain",
        "Raindrop"
    ],

    "rooster": [
        "Rooster",
        "Cock-a-doodle-doo",
        "Chicken, rooster"
    ],

    "sheep": [
        "Sheep",
        "Bleat"
    ],

    "siren": [
        "Siren"
    ],

    "thunderstorm": [
        "Thunderstorm",
        "Thunder"
    ],

    "water_drops": [
        "Drip",
        "Raindrop",
        "Water"
    ],

    "wind": [
        "Wind"
    ]
}


# ============================================================
# HELPER: LOAD AUDIO WINDOW
# ============================================================

def load_audio_window(audio_path, start, end):

    duration = max(
        end - start,
        0
    )

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start,
        duration=duration
    )

    target_samples = int(
        WINDOW_SECONDS * SAMPLE_RATE
    )

    # Pad short final windows
    if len(audio) < target_samples:

        padding = target_samples - len(audio)

        audio = np.pad(
            audio,
            (0, padding)
        )

    # Trim anything longer than 5 seconds
    elif len(audio) > target_samples:

        audio = audio[:target_samples]

    return audio.astype(
        np.float32
    )


# ============================================================
# HELPER: NORMALIZE LABEL
# ============================================================

def normalize_label(label):

    return (
        str(label)
        .strip()
        .lower()
    )


# ============================================================
# BUILD AST LABEL MAPPING
# ============================================================

def build_label_mapping(model):

    id2label = model.config.id2label

    normalized_model_labels = {
        normalize_label(label): int(index)
        for index, label in id2label.items()
    }

    resolved_mapping = {}

    print("\n" + "=" * 60)
    print("AST LABEL MAPPING")
    print("=" * 60)

    for target_class in TARGET_CLASSES:

        candidates = LABEL_MAPPING.get(
            target_class,
            []
        )

        matches = []

        for candidate in candidates:

            candidate_normalized = normalize_label(
                candidate
            )

            if candidate_normalized in normalized_model_labels:

                index = normalized_model_labels[
                    candidate_normalized
                ]

                matches.append(
                    {
                        "label": id2label[index],
                        "index": index
                    }
                )

        # Remove duplicate AudioSet indices
        unique_matches = []

        seen_indices = set()

        for match in matches:

            if match["index"] not in seen_indices:

                unique_matches.append(
                    match
                )

                seen_indices.add(
                    match["index"]
                )

        resolved_mapping[
            target_class
        ] = unique_matches

        if unique_matches:

            labels = [
                match["label"]
                for match in unique_matches
            ]

            print(
                f"{target_class:<22} -> {labels}"
            )

        else:

            print(
                f"{target_class:<22} -> NO MATCH"
            )

    return resolved_mapping


# ============================================================
# CONVERT AUDIOSET OUTPUTS TO OUR 20 CLASSES
# ============================================================

def get_target_probabilities(
    probabilities,
    resolved_mapping
):

    target_probabilities = []

    for target_class in TARGET_CLASSES:

        matches = resolved_mapping[
            target_class
        ]

        if not matches:

            target_probabilities.append(
                0.0
            )

            continue

        scores = []

        for match in matches:

            score = probabilities[
                match["index"]
            ]

            scores.append(
                float(score)
            )

        # If multiple AudioSet labels map
        # to one target class, use maximum.
        target_probabilities.append(
            max(scores)
        )

    return np.array(
        target_probabilities,
        dtype=np.float32
    )


# ============================================================
# CONVERT GROUND TRUTH CLASS NAMES TO VECTOR
# ============================================================

def get_ground_truth_vector(window):

    # The window index stores class names such as:
    #
    # ["engine", "footsteps", "wind"]
    #
    # Convert them into:
    #
    # [0, 0, ..., 1, 1, ..., 1]

    window_labels = set(
        window["labels"]
    )

    return np.array(
        [
            1 if class_name in window_labels else 0
            for class_name in TARGET_CLASSES
        ],
        dtype=int
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AUDIO CONTEXT LAYER - AST EVALUATION")
    print("=" * 60)

    os.makedirs(
        "results",
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD WINDOW INDEX
    # --------------------------------------------------------

    print("\nLoading window index...")

    with open(
        WINDOW_INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        window_data = json.load(f)

    windows = window_data["windows"]

    test_windows = [
        window
        for window in windows
        if window["split"] == "test"
    ]

    print(
        f"Total windows: {len(windows)}"
    )

    print(
        f"Test windows: {len(test_windows)}"
    )

    # --------------------------------------------------------
    # LOAD FEATURE EXTRACTOR
    # --------------------------------------------------------

    print("\nLoading AST feature extractor...")

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading AST model...")

    model = (
        AutoModelForAudioClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()

    print(
        f"AST AudioSet classes: "
        f"{len(model.config.id2label)}"
    )

    # --------------------------------------------------------
    # BUILD LABEL MAPPING
    # --------------------------------------------------------

    resolved_mapping = build_label_mapping(
        model
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    prediction_records = []

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RUNNING AST ON TEST WINDOWS")
    print("=" * 60)

    total_windows = len(
        test_windows
    )

    for counter, window in enumerate(
        test_windows,
        start=1
    ):

        audio_path = window["audio_path"]

        start = float(
            window["start"]
        )

        end = float(
            window["end"]
        )

        # ----------------------------------------------------
        # LOAD 5 SECOND AUDIO WINDOW
        # ----------------------------------------------------

        audio = load_audio_window(
            audio_path,
            start,
            end
        )

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        # ----------------------------------------------------
        # AST INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            probabilities = torch.sigmoid(
                outputs.logits
            )[0].cpu().numpy()

        # ----------------------------------------------------
        # MAP AUDIOSET → TARGET CLASSES
        # ----------------------------------------------------

        target_probabilities = (
            get_target_probabilities(
                probabilities,
                resolved_mapping
            )
        )

        predicted_labels = (
            target_probabilities >= THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        ground_truth_labels = (
            get_ground_truth_vector(
                window
            )
        )

        # ----------------------------------------------------
        # STORE METRICS ARRAYS
        # ----------------------------------------------------

        y_true.append(
            ground_truth_labels
        )

        y_pred.append(
            predicted_labels
        )

        # ----------------------------------------------------
        # STORE WINDOW PREDICTION
        # ----------------------------------------------------

        prediction_records.append(
            {
                "scene_id": window["scene_id"],

                "audio_path": audio_path,

                "start": start,

                "end": end,

                "ground_truth": {
                    class_name: int(value)
                    for class_name, value in zip(
                        TARGET_CLASSES,
                        ground_truth_labels
                    )
                },

                "probabilities": {
                    class_name: float(probability)
                    for class_name, probability in zip(
                        TARGET_CLASSES,
                        target_probabilities
                    )
                },

                "predictions": {
                    class_name: int(value)
                    for class_name, value in zip(
                        TARGET_CLASSES,
                        predicted_labels
                    )
                }
            }
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            counter == 1
            or counter % 25 == 0
            or counter == total_windows
        ):

            print(
                f"Processed "
                f"{counter}/{total_windows} "
                f"test windows"
            )

    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    y_true = np.array(
        y_true,
        dtype=int
    )

    y_pred = np.array(
        y_pred,
        dtype=int
    )

    print("\n" + "=" * 60)
    print("CALCULATING METRICS")
    print("=" * 60)

    print(
        f"Ground truth shape: "
        f"{y_true.shape}"
    )

    print(
        f"Prediction shape: "
        f"{y_pred.shape}"
    )

    # ========================================================
    # MICRO METRICS
    # ========================================================

    (
        micro_precision,
        micro_recall,
        micro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    # ========================================================
    # MACRO METRICS
    # ========================================================

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # PER CLASS METRICS
    # ========================================================

    (
        class_precision,
        class_recall,
        class_f1,
        class_support
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print("\n--- AST TEST RESULTS ---")

    print(
        f"Threshold:       {THRESHOLD:.2f}"
    )

    print(
        f"Test windows:    {len(test_windows)}"
    )

    print(
        f"Micro Precision: {micro_precision:.4f}"
    )

    print(
        f"Micro Recall:    {micro_recall:.4f}"
    )

    print(
        f"Micro F1:        {micro_f1:.4f}"
    )

    print(
        f"Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"Macro Recall:    {macro_recall:.4f}"
    )

    print(
        f"Macro F1:        {macro_f1:.4f}"
    )

    # ========================================================
    # PER CLASS RESULTS
    # ========================================================

    print("\n--- PER CLASS RESULTS ---")

    print(
        f"{'Class':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print("-" * 70)

    per_class_results = []

    for index, class_name in enumerate(
        TARGET_CLASSES
    ):

        result = {
            "class": class_name,

            "precision": float(
                class_precision[index]
            ),

            "recall": float(
                class_recall[index]
            ),

            "f1": float(
                class_f1[index]
            ),

            "support": int(
                class_support[index]
            )
        }

        per_class_results.append(
            result
        )

        print(
            f"{class_name:<22}"
            f"{result['precision']:>12.3f}"
            f"{result['recall']:>12.3f}"
            f"{result['f1']:>12.3f}"
            f"{result['support']:>12}"
        )

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics = {

        "model": MODEL_NAME,

        "sample_rate": SAMPLE_RATE,

        "window_seconds": WINDOW_SECONDS,

        "threshold": THRESHOLD,

        "test_windows": len(
            test_windows
        ),

        "target_classes": TARGET_CLASSES,

        "micro": {
            "precision": float(
                micro_precision
            ),
            "recall": float(
                micro_recall
            ),
            "f1": float(
                micro_f1
            )
        },

        "macro": {
            "precision": float(
                macro_precision
            ),
            "recall": float(
                macro_recall
            ),
            "f1": float(
                macro_f1
            )
        },

        "per_class": per_class_results,

        "label_mapping": resolved_mapping
    }

    with open(
        OUTPUT_METRICS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ========================================================
    # SAVE PER CLASS CSV
    # ========================================================

    with open(
        OUTPUT_PER_CLASS,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "precision",
                "recall",
                "f1",
                "support"
            ]
        )

        writer.writeheader()

        writer.writerows(
            per_class_results
        )

    # ========================================================
    # SAVE WINDOW PREDICTIONS
    # ========================================================

    with open(
        OUTPUT_PREDICTIONS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prediction_records,
            f,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("AST EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"\nMetrics:"
        f"\n{OUTPUT_METRICS}"
    )

    print(
        f"\nPer class:"
        f"\n{OUTPUT_PER_CLASS}"
    )

    print(
        f"\nWindow predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()