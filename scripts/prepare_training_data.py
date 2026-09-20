import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

MANIFEST_DIR = Path("data/manifests")
AUDIO_DIR = Path("data/scenes")

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "window_index.json"

WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5

# An event must overlap the window by at least this much
# to be considered present in that window.
MIN_OVERLAP_SECONDS = 0.5


# ============================================================
# LOAD MANIFESTS
# ============================================================

def load_manifests():

    manifests = []

    for path in sorted(
        MANIFEST_DIR.glob("scene_*.json")
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            scene = json.load(f)

        manifests.append(scene)

    return manifests


# ============================================================
# GET ALL LABELS
# ============================================================

def get_all_labels(manifests):

    labels = set()

    for scene in manifests:

        for event in scene["events"]:

            labels.add(event["label"])

    return sorted(labels)


# ============================================================
# CALCULATE OVERLAP
# ============================================================

def calculate_overlap(
    window_start,
    window_end,
    event_start,
    event_end
):

    overlap_start = max(
        window_start,
        event_start
    )

    overlap_end = min(
        window_end,
        event_end
    )

    return max(
        0.0,
        overlap_end - overlap_start
    )


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_scene_windows(
    scene,
    label_to_index
):

    duration = float(
        scene["duration"]
    )

    windows = []

    start = 0.0

    while start < duration:

        end = min(
            start + WINDOW_SECONDS,
            duration
        )

        labels = []

        event_details = []

        for event in scene["events"]:

            overlap = calculate_overlap(
                start,
                end,
                float(event["start"]),
                float(event["end"])
            )

            if overlap >= MIN_OVERLAP_SECONDS:

                label = event["label"]

                if label not in labels:
                    labels.append(label)

                event_details.append({
                    "event_id": event["event_id"],
                    "label": label,
                    "overlap": round(
                        overlap,
                        3
                    )
                })

        # ----------------------------------------------------
        # Multi hot target vector
        # ----------------------------------------------------

        target = [
            0
            for _ in range(
                len(label_to_index)
            )
        ]

        for label in labels:

            target[
                label_to_index[label]
            ] = 1

        audio_path = (
            AUDIO_DIR
            / f"{scene['scene_id']}.wav"
        )

        if not audio_path.exists():

            print(
                f"WARNING: Missing audio "
                f"{audio_path}"
            )

        windows.append({

            "scene_id": scene["scene_id"],

            "split": scene["split"],

            "audio_path": str(
                audio_path
            ),

            "start": round(
                start,
                3
            ),

            "end": round(
                end,
                3
            ),

            "labels": sorted(
                labels
            ),

            "target": target,

            "event_details":
                event_details
        })

        start += HOP_SECONDS

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO CONTEXT LAYER")
    print("TRAINING WINDOW PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load scenes
    # --------------------------------------------------------

    print("\nLoading manifests...")

    manifests = load_manifests()

    print(
        f"Loaded {len(manifests)} scenes."
    )

    if not manifests:

        raise RuntimeError(
            "No scene manifests found."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = get_all_labels(
        manifests
    )

    label_to_index = {
        label: index
        for index, label in enumerate(
            labels
        )
    }

    print(
        f"\nFound {len(labels)} sound labels."
    )

    print("\nLabels:")

    for index, label in enumerate(
        labels
    ):

        print(
            f"{index:02d}  {label}"
        )

    # --------------------------------------------------------
    # Generate windows
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING WINDOWS")
    print("=" * 70)

    all_windows = []

    for scene_number, scene in enumerate(
        manifests,
        start=1
    ):

        scene_windows = create_scene_windows(
            scene,
            label_to_index
        )

        all_windows.extend(
            scene_windows
        )

        print(
            f"[{scene_number:03d}/"
            f"{len(manifests):03d}] "
            f"{scene['scene_id']} "
            f"({scene['split']}) -> "
            f"{len(scene_windows)} windows"
        )

    # --------------------------------------------------------
    # Split statistics
    # --------------------------------------------------------

    split_counts = {}

    for window in all_windows:

        split = window["split"]

        split_counts[split] = (
            split_counts.get(split, 0)
            + 1
        )

    print("\n")
    print("=" * 70)
    print("WINDOW COUNTS")
    print("=" * 70)

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"{split:<15} {count}"
        )

    # --------------------------------------------------------
    # Label statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LABEL STATISTICS")
    print("=" * 70)

    for label in labels:

        count = sum(
            1
            for window in all_windows
            if label in window["labels"]
        )

        print(
            f"{label:<25} {count}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "window_seconds":
            WINDOW_SECONDS,

        "hop_seconds":
            HOP_SECONDS,

        "min_overlap_seconds":
            MIN_OVERLAP_SECONDS,

        "num_scenes":
            len(manifests),

        "num_windows":
            len(all_windows),

        "labels":
            labels,

        "label_to_index":
            label_to_index,

        "windows":
            all_windows
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print("\n")
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal scenes: "
        f"{len(manifests)}"
    )

    print(
        f"Total windows: "
        f"{len(all_windows)}"
    )

    print(
        f"Number of labels: "
        f"{len(labels)}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()