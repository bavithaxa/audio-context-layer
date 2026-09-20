import json
import os

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from panns_inference import AudioTagging, labels


# ============================================================
# Configuration
# ============================================================

INDEX_PATH = "data/processed/window_index.json"

CHECKPOINT_PATH = r"C:\Users\ASUS\panns_data\Cnn14_mAP=0.431.pth"

RESULTS_DIR = "results"

SAMPLE_RATE = 32000

WINDOW_SECONDS = 5.0

WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

THRESHOLD = 0.30

BATCH_SIZE = 8


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
    "wind",
]


# ============================================================
# PANNs AudioSet label mapping
# ============================================================

PANN_LABEL_MAP = {

    "car_horn": [
        "Car horn",
        "Car horn, honking",
        "Honk",
    ],

    "chirping_birds": [
        "Chirp, tweet",
        "Bird",
        "Bird vocalization, bird call, bird song",
    ],

    "clock_tick": [
        "Tick",
        "Tick-tock",
        "Clock",
    ],

    "cow": [
        "Cow",
        "Moo",
    ],

    "crow": [
        "Crow",
        "Caw",
    ],

    "dog": [
        "Dog",
        "Bark",
        "Bow-wow",
        "Whimper (dog)",
        "Yip",
    ],

    "door_wood_knock": [
        "Knock",
        "Door",
        "Wood",
    ],

    "engine": [
        "Engine",
        "Idling",
        "Vehicle",
    ],

    "footsteps": [
        "Walk, footsteps",
        "Footstep",
    ],

    "hen": [
        "Chicken, rooster",
        "Chicken",
        "Cluck",
    ],

    "keyboard_typing": [
        "Typing",
        "Computer keyboard",
    ],

    "laughing": [
        "Laughter",
    ],

    "mouse_click": [
        "Mouse click",
        "Click",
    ],

    "rain": [
        "Rain",
        "Raindrop",
    ],

    "rooster": [
        "Rooster",
        "Cock-a-doodle-doo",
        "Chicken, rooster",
    ],

    "sheep": [
        "Sheep",
        "Bleat",
    ],

    "siren": [
        "Siren",
    ],

    "thunderstorm": [
        "Thunderstorm",
        "Thunder",
    ],

    "water_drops": [
        "Drip",
        "Raindrop",
        "Water",
    ],

    "wind": [
        "Wind",
    ],
}


# ============================================================
# Load window index
# ============================================================

def load_window_index():

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "windows" in data:
            return data["windows"]

        if "items" in data:
            return data["items"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Could not find window list in window_index.json"
    )


# ============================================================
# Resolve PANNs labels
# ============================================================

def resolve_mapping(label_names):

    normalized = {
        name.lower(): index
        for index, name in enumerate(label_names)
    }

    resolved = {}

    for target in TARGET_CLASSES:

        matches = []

        for candidate in PANN_LABEL_MAP.get(
            target,
            []
        ):

            candidate_lower = candidate.lower()

            if candidate_lower in normalized:

                matches.append(
                    normalized[candidate_lower]
                )

        resolved[target] = matches

    return resolved


# ============================================================
# Load audio window
# ============================================================

def load_audio_window(
    audio_cache,
    audio_path,
    start,
    end
):

    # --------------------------------------------------------
    # Load complete scene only once
    # --------------------------------------------------------

    if audio_path not in audio_cache:

        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        audio_cache[audio_path] = audio

    audio = audio_cache[audio_path]

    # --------------------------------------------------------
    # Convert timestamps to samples
    # --------------------------------------------------------

    start_sample = int(
        start * SAMPLE_RATE
    )

    end_sample = int(
        end * SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Extract actual audio
    # --------------------------------------------------------

    window = audio[
        start_sample:end_sample
    ]

    # --------------------------------------------------------
    # ALWAYS make the model input exactly
    # 5 seconds = 160,000 samples
    #
    # This is important because some final windows
    # are shorter than 5 seconds.
    # --------------------------------------------------------

    if len(window) < WINDOW_SAMPLES:

        padding_amount = (
            WINDOW_SAMPLES - len(window)
        )

        window = np.pad(
            window,
            (
                0,
                padding_amount
            ),
            mode="constant"
        )

    elif len(window) > WINDOW_SAMPLES:

        window = window[
            :WINDOW_SAMPLES
        ]

    return window.astype(
        np.float32
    )


# ============================================================
# Find audio path
# ============================================================

def get_audio_path(window_info):

    # First try explicit audio path

    possible_paths = [

        window_info.get(
            "audio_path",
            ""
        ),

        window_info.get(
            "audio",
            ""
        ),
    ]

    for path in possible_paths:

        if path and os.path.exists(path):

            return path

    # Otherwise use scene ID

    scene_id = window_info.get(
        "scene_id"
    )

    if scene_id:

        scene_path = (
            f"data/scenes/{scene_id}.wav"
        )

        if os.path.exists(scene_path):

            return scene_path

    raise FileNotFoundError(
        f"Could not find audio for "
        f"scene {scene_id}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PANNs CNN14 Evaluation")
    print("=" * 70)

    print(
        f"\nPANNs input window: "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        f"PANNs input samples: "
        f"{WINDOW_SAMPLES}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD}"
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # Load window index
    # ========================================================

    windows = load_window_index()

    test_windows = [
        window
        for window in windows
        if window.get("split") == "test"
    ]

    print(
        f"\nTotal windows: "
        f"{len(windows)}"
    )

    print(
        f"Test windows: "
        f"{len(test_windows)}"
    )

    if not test_windows:

        raise ValueError(
            "No test windows found."
        )

    # ========================================================
    # Load PANNs labels
    # ========================================================

    label_names = list(labels)

    print(
        f"PANNs AudioSet classes: "
        f"{len(label_names)}"
    )

    mapping = resolve_mapping(
        label_names
    )

    # ========================================================
    # Print mapping
    # ========================================================

    print(
        "\nTarget label mapping:"
    )

    print(
        "-" * 70
    )

    for target in TARGET_CLASSES:

        indices = mapping[target]

        matched_names = [
            label_names[index]
            for index in indices
        ]

        if matched_names:

            print(
                f"{target:<22} -> "
                f"{matched_names}"
            )

        else:

            print(
                f"{target:<22} -> "
                f"NO MATCH"
            )

    # ========================================================
    # Load model
    # ========================================================

    print(
        "\nLoading PANNs CNN14..."
    )

    model = AudioTagging(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    )

    print(
        "Model loaded."
    )

    # ========================================================
    # Storage
    # ========================================================

    y_true = []

    y_pred = []

    prediction_records = []

    audio_cache = {}

    total = len(test_windows)

    # ========================================================
    # Process batches
    # ========================================================

    for batch_start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch_windows = test_windows[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_audio = []

        # ----------------------------------------------------
        # Load windows
        # ----------------------------------------------------

        for window_info in batch_windows:

            audio_path = get_audio_path(
                window_info
            )

            audio_window = load_audio_window(
                audio_cache,
                audio_path,
                window_info["start"],
                window_info["end"]
            )

            batch_audio.append(
                audio_window
            )

        # ----------------------------------------------------
        # Convert to batch
        # ----------------------------------------------------

        batch_audio = np.stack(
            batch_audio,
            axis=0
        )

        # ----------------------------------------------------
        # PANNs inference
        # ----------------------------------------------------

        clipwise_output, _ = model.inference(
            batch_audio
        )

        # ----------------------------------------------------
        # Process each window
        # ----------------------------------------------------

        for local_index, window_info in enumerate(
            batch_windows
        ):

            scores = clipwise_output[
                local_index
            ]

            target_scores = []

            # ------------------------------------------------
            # Map AudioSet → ESC 50 target classes
            # ------------------------------------------------

            for target in TARGET_CLASSES:

                indices = mapping[target]

                if indices:

                    score = max(
                        float(scores[index])
                        for index in indices
                    )

                else:

                    score = 0.0

                target_scores.append(
                    score
                )

            target_scores = np.array(
                target_scores,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            predictions = (
                target_scores >= THRESHOLD
            ).astype(int)

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            true_labels = set(
                window_info.get(
                    "labels",
                    []
                )
            )

            true_vector = np.array(
                [
                    int(
                        label in true_labels
                    )
                    for label in TARGET_CLASSES
                ],
                dtype=int
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            y_true.append(
                true_vector
            )

            y_pred.append(
                predictions
            )

            prediction_records.append({

                "scene_id":
                    window_info["scene_id"],

                "start":
                    window_info["start"],

                "end":
                    window_info["end"],

                "true_labels":
                    sorted(
                        list(true_labels)
                    ),

                "predicted_labels":
                    [
                        TARGET_CLASSES[i]
                        for i, value
                        in enumerate(
                            predictions
                        )
                        if value == 1
                    ],

                "scores":
                    {
                        TARGET_CLASSES[i]:
                            float(
                                target_scores[i]
                            )
                        for i in range(
                            len(TARGET_CLASSES)
                        )
                    }
            })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = min(
            batch_start + BATCH_SIZE,
            total
        )

        print(
            f"Processed "
            f"{processed}/{total}"
        )

    # ========================================================
    # Convert to NumPy
    # ========================================================

    y_true = np.array(
        y_true,
        dtype=int
    )

    y_pred = np.array(
        y_pred,
        dtype=int
    )

    print(
        "\nPrediction matrix shapes:"
    )

    print(
        f"True: {y_true.shape}"
    )

    print(
        f"Pred: {y_pred.shape}"
    )

    # ========================================================
    # Overall metrics
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
    # Print overall results
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Threshold:          "
        f"{THRESHOLD}"
    )

    print(
        f"Micro Precision:    "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall:       "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1:            "
        f"{micro_f1:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:            "
        f"{macro_f1:.4f}"
    )

    # ========================================================
    # Per class metrics
    # ========================================================

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class = []

    print(
        "\n" + "=" * 70
    )

    print(
        "PER CLASS RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Class':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print(
        "-" * 70
    )

    for i, class_name in enumerate(
        TARGET_CLASSES
    ):

        row = {

            "class":
                class_name,

            "precision":
                float(
                    precision[i]
                ),

            "recall":
                float(
                    recall[i]
                ),

            "f1":
                float(
                    f1[i]
                ),

            "support":
                int(
                    support[i]
                ),
        }

        per_class.append(
            row
        )

        print(
            f"{class_name:<22}"
            f"{precision[i]:>12.3f}"
            f"{recall[i]:>12.3f}"
            f"{f1[i]:>12.3f}"
            f"{support[i]:>12}"
        )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics = {

        "model":
            "PANNs CNN14",

        "checkpoint":
            "Cnn14_mAP=0.431.pth",

        "split":
            "test",

        "num_test_windows":
            int(
                len(test_windows)
            ),

        "input_window_seconds":
            WINDOW_SECONDS,

        "threshold":
            THRESHOLD,

        "micro_precision":
            float(
                micro_precision
            ),

        "micro_recall":
            float(
                micro_recall
            ),

        "micro_f1":
            float(
                micro_f1
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "label_mapping":
            {
                target:
                    [
                        label_names[index]
                        for index in mapping[target]
                    ]
                for target in TARGET_CLASSES
            }
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "panns_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ========================================================
    # Save per class CSV
    # ========================================================

    per_class_path = os.path.join(
        RESULTS_DIR,
        "panns_per_class.csv"
    )

    pd.DataFrame(
        per_class
    ).to_csv(
        per_class_path,
        index=False
    )

    # ========================================================
    # Save window predictions
    # ========================================================

    predictions_path = os.path.join(
        RESULTS_DIR,
        "panns_window_predictions.json"
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

    # ========================================================
    # Final output
    # ========================================================

    print(
        "\nResults saved:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {per_class_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PANNs evaluation complete"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()