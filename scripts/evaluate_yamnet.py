import os
import json
import csv

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics import (
    precision_recall_fscore_support
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WINDOW_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "window_index.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# Configuration
# =========================================================

TARGET_SR = 16000

THRESHOLD = 0.30


# =========================================================
# Target classes
# =========================================================

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


# =========================================================
# YAMNet mapping
# =========================================================

YAMNET_MAPPING = {

    "car_horn": [
        "Honk",
        "Vehicle horn, car horn, honking"
    ],

    "chirping_birds": [
        "Bird",
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet"
    ],

    "clock_tick": [
        "Tick-tock",
        "Clock"
    ],

    "cow": [
        "Moo",
        "Cowbell"
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
        "Growling",
        "Howl",
        "Yip"
    ],

    "door_wood_knock": [
        "Knock",
        "Door"
    ],

    "engine": [
        "Engine",
        "Accelerating, revving, vroom"
    ],

    "footsteps": [
        "Walk, footsteps",
        "Footsteps"
    ],

    "hen": [
        "Chicken",
        "Cluck"
    ],

    "keyboard_typing": [
        "Typing",
        "Computer keyboard"
    ],

    "laughing": [
        "Laughter",
        "Giggle"
    ],

    "mouse_click": [
        "Mouse click"
    ],

    "rain": [
        "Rain"
    ],

    "rooster": [
        "Rooster",
        "Cock-a-doodle-doo"
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
        "Water",
        "Liquid"
    ],

    "wind": [
        "Wind"
    ]
}


# =========================================================
# Load YAMNet class names
# =========================================================

def load_yamnet_class_names(model):

    class_map_path = (
        model
        .class_map_path()
        .numpy()
        .decode("utf-8")
    )

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

    return class_names


# =========================================================
# Build class mapping
# =========================================================

def build_mapping_indices(class_names):

    name_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    mapping_indices = {}

    print("\nYAMNet mapping:")

    for target_class in TARGET_CLASSES:

        indices = []

        for yamnet_name in YAMNET_MAPPING[
            target_class
        ]:

            if yamnet_name in name_to_index:

                indices.append(
                    name_to_index[yamnet_name]
                )

        mapping_indices[target_class] = indices

        print(
            f"{target_class:22s} -> "
            f"{len(indices)} YAMNet classes"
        )

        if len(indices) == 0:

            print(
                "  WARNING: no matching YAMNet "
                "class found"
            )

    return mapping_indices


# =========================================================
# Load audio
# =========================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    return audio.astype(np.float32)


# =========================================================
# Run YAMNet
# =========================================================

def run_yamnet(model, audio):

    waveform = tf.convert_to_tensor(
        audio,
        dtype=tf.float32
    )

    scores, _, _ = model(waveform)

    return scores.numpy()


# =========================================================
# Aggregate YAMNet frames
# =========================================================

def aggregate_window_scores(
    frame_scores,
    start_time,
    end_time,
    mapping_indices
):

    num_frames = frame_scores.shape[0]

    frame_times = (
        np.arange(num_frames) * 0.48
        + 0.48
    )

    frame_mask = (
        (frame_times >= start_time)
        &
        (frame_times <= end_time)
    )

    selected_scores = frame_scores[
        frame_mask
    ]

    if selected_scores.shape[0] == 0:

        selected_scores = frame_scores

    target_scores = {}

    for target_class in TARGET_CLASSES:

        indices = mapping_indices[
            target_class
        ]

        if not indices:

            target_scores[target_class] = 0.0
            continue

        class_scores = selected_scores[
            :,
            indices
        ]

        target_scores[target_class] = float(
            np.max(class_scores)
        )

    return target_scores


# =========================================================
# Load window index
# =========================================================

print("Loading window index...")

with open(
    WINDOW_INDEX_PATH,
    "r",
    encoding="utf-8"
) as f:

    window_data = json.load(f)


test_windows = [
    window
    for window in window_data["windows"]
    if window["split"] == "test"
]

print(
    f"Total windows: "
    f"{len(window_data['windows'])}"
)

print(
    f"Test windows: "
    f"{len(test_windows)}"
)


# =========================================================
# Load YAMNet
# =========================================================

print("\nLoading YAMNet...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet loaded.")


# =========================================================
# Load classes
# =========================================================

class_names = load_yamnet_class_names(
    yamnet_model
)

print(
    f"Loaded {len(class_names)} YAMNet classes."
)


# =========================================================
# Build mapping
# =========================================================

mapping_indices = build_mapping_indices(
    class_names
)


# =========================================================
# Group windows by scene
# =========================================================

scene_windows = {}

for window in test_windows:

    scene_id = window["scene_id"]

    if scene_id not in scene_windows:

        scene_windows[scene_id] = []

    scene_windows[scene_id].append(
        window
    )


print(
    f"\nTest scenes: "
    f"{len(scene_windows)}"
)


# =========================================================
# Store predictions as matrices
# =========================================================

all_true = []
all_pred = []

prediction_records = []


# =========================================================
# Run evaluation
# =========================================================

print("\nRunning YAMNet evaluation...")


for scene_number, (
    scene_id,
    windows
) in enumerate(
    scene_windows.items(),
    start=1
):

    scene_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "scenes",
        f"{scene_id}.wav"
    )

    print(
        f"[{scene_number}/"
        f"{len(scene_windows)}] "
        f"{scene_id}"
    )

    audio = load_audio(
        scene_path
    )

    frame_scores = run_yamnet(
        yamnet_model,
        audio
    )

    for window in windows:

        start_time = window["start"]
        end_time = window["end"]

        scores = aggregate_window_scores(
            frame_scores,
            start_time,
            end_time,
            mapping_indices
        )

        true_vector = []
        pred_vector = []

        predicted_labels = []
        true_labels = window["labels"]

        for label in TARGET_CLASSES:

            true_value = (
                1
                if label in true_labels
                else 0
            )

            predicted_value = (
                1
                if scores[label] >= THRESHOLD
                else 0
            )

            true_vector.append(
                true_value
            )

            pred_vector.append(
                predicted_value
            )

            if predicted_value == 1:

                predicted_labels.append(
                    label
                )

        all_true.append(
            true_vector
        )

        all_pred.append(
            pred_vector
        )

        prediction_records.append({
            "scene_id": scene_id,
            "window_start": start_time,
            "window_end": end_time,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "scores": scores
        })


# =========================================================
# Convert to matrices
# =========================================================

y_true = np.asarray(
    all_true,
    dtype=np.int32
)

y_pred = np.asarray(
    all_pred,
    dtype=np.int32
)


print("\nEvaluation matrix:")
print(
    f"Samples: {y_true.shape[0]}"
)

print(
    f"Classes: {y_true.shape[1]}"
)


# =========================================================
# Correct multilabel metrics
# =========================================================

micro_precision, micro_recall, micro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)


# =========================================================
# Per class metrics
# =========================================================

per_precision, per_recall, per_f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
)


# =========================================================
# Print overall results
# =========================================================

print("\n" + "=" * 70)
print("YAMNet OVERALL RESULTS")
print("=" * 70)

print(
    f"Threshold:       {THRESHOLD:.2f}"
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


# =========================================================
# Print per class results
# =========================================================

print("\n" + "=" * 70)
print("PER CLASS RESULTS")
print("=" * 70)


per_class_results = {}

for index, label in enumerate(
    TARGET_CLASSES
):

    print(
        f"{label:22s} "
        f"P={per_precision[index]:.3f} "
        f"R={per_recall[index]:.3f} "
        f"F1={per_f1[index]:.3f} "
        f"Support={support[index]}"
    )

    per_class_results[label] = {

        "precision": float(
            per_precision[index]
        ),

        "recall": float(
            per_recall[index]
        ),

        "f1": float(
            per_f1[index]
        ),

        "support": int(
            support[index]
        )
    }


# =========================================================
# Save metrics
# =========================================================

metrics = {

    "model": "YAMNet",

    "threshold": THRESHOLD,

    "test_windows": int(
        y_true.shape[0]
    ),

    "test_scenes": len(
        scene_windows
    ),

    "overall": {

        "micro_precision": float(
            micro_precision
        ),

        "micro_recall": float(
            micro_recall
        ),

        "micro_f1": float(
            micro_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        )
    },

    "per_class": per_class_results
}


json_path = os.path.join(
    RESULTS_DIR,
    "yamnet_metrics.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# =========================================================
# Save CSV
# =========================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "yamnet_per_class.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "precision",
        "recall",
        "f1",
        "support"
    ])

    for label in TARGET_CLASSES:

        result = per_class_results[label]

        writer.writerow([
            label,
            result["precision"],
            result["recall"],
            result["f1"],
            result["support"]
        ])


# =========================================================
# Save predictions
# =========================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "yamnet_window_predictions.json"
)

with open(
    predictions_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        prediction_records,
        f,
        indent=2
    )


# =========================================================
# Final
# =========================================================

print("\n" + "=" * 70)
print("Saved results")
print("=" * 70)

print(
    f"Metrics:       {json_path}"
)

print(
    f"Per class CSV: {csv_path}"
)

print(
    f"Predictions:   {predictions_path}"
)

print("\nDone.")