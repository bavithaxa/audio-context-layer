import json
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# CONFIG
# ============================================================

MODEL_URL = "https://tfhub.dev/google/yamnet/1"

TARGET_SR = 16000

# Keep these fixed for this experiment.
THRESHOLD = 0.30
MIN_EVENT_DURATION = 0.40

# Changed from 0.75 to 1.5 seconds.
# This allows short confidence dips inside one continuous event
# to be merged back into the same event.
MAX_GAP = 1.5


# ============================================================
# TARGET LABEL MAPPING
# ============================================================

TARGET_TO_YAMNET = {

    "car_horn": [
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
    ],

    "hen": [
        "Chicken, rooster",
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
        # No reliable YAMNet mapping.
    ],

    "rain": [
        "Rain",
        "Raindrop",
    ],

    "rooster": [
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
# LOAD YAMNET
# ============================================================

print("Loading YAMNet...")

yamnet_model = hub.load(MODEL_URL)

print("YAMNet loaded.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/"
    "tensorflow/models/master/research/audioset/"
    "yamnet/yamnet_class_map.csv"
)


def load_class_names():

    import urllib.request
    import csv
    import io

    with urllib.request.urlopen(
        CLASS_MAP_URL
    ) as response:

        content = response.read().decode(
            "utf-8"
        )

    reader = csv.DictReader(
        io.StringIO(content)
    )

    return [
        row["display_name"]
        for row in reader
    ]


CLASS_NAMES = load_class_names()

print(
    f"Loaded {len(CLASS_NAMES)} YAMNet classes."
)


# ============================================================
# LABEL INDEX LOOKUP
# ============================================================

YAMNET_INDEX = {
    name: index
    for index, name in enumerate(
        CLASS_NAMES
    )
}


TARGET_INDEXES = {}

for target, yamnet_labels in TARGET_TO_YAMNET.items():

    indexes = []

    for label in yamnet_labels:

        if label in YAMNET_INDEX:

            indexes.append(
                YAMNET_INDEX[label]
            )

    TARGET_INDEXES[target] = indexes


print("\nTarget label mapping:")

for target, indexes in TARGET_INDEXES.items():

    labels = [
        CLASS_NAMES[i]
        for i in indexes
    ]

    print(
        f"  {target}: {labels}"
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    print(
        f"\nLoading audio: {audio_path}"
    )

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    audio = audio.astype(
        np.float32
    )

    duration = (
        len(audio)
        / TARGET_SR
    )

    print(
        f"Sample rate: {TARGET_SR} Hz"
    )

    print(
        f"Duration: {duration:.2f} sec"
    )

    return audio, duration


# ============================================================
# RUN YAMNET
# ============================================================

def run_yamnet(audio):

    waveform = tf.convert_to_tensor(
        audio,
        dtype=tf.float32
    )

    scores, embeddings, spectrogram = (
        yamnet_model(waveform)
    )

    scores = scores.numpy()

    print(
        f"YAMNet frames: {scores.shape[0]}"
    )

    print(
        f"YAMNet classes: {scores.shape[1]}"
    )

    return scores


# ============================================================
# TARGET SCORE PER FRAME
# ============================================================

def get_target_scores(
    scores,
    target
):

    indexes = TARGET_INDEXES.get(
        target,
        []
    )

    if not indexes:

        return np.zeros(
            scores.shape[0],
            dtype=np.float32
        )

    return np.max(
        scores[:, indexes],
        axis=1
    )


# ============================================================
# FRAME TIMING
# ============================================================

# YAMNet uses approximately:
# frame duration = 0.96 sec
# frame hop      = 0.48 sec

FRAME_DURATION = 0.96
FRAME_HOP = 0.48


def get_frame_times(
    frame_index,
    audio_duration
):

    start = (
        frame_index
        * FRAME_HOP
    )

    end = min(
        start + FRAME_DURATION,
        audio_duration
    )

    return start, end


# ============================================================
# FIND ACTIVE REGIONS
# ============================================================

def find_active_regions(
    frame_scores,
    audio_duration
):

    active = (
        frame_scores
        >= THRESHOLD
    )

    regions = []

    start_frame = None

    for i, is_active in enumerate(active):

        if is_active and start_frame is None:

            start_frame = i

        elif (
            not is_active
            and start_frame is not None
        ):

            end_frame = i - 1

            regions.append(
                (
                    start_frame,
                    end_frame
                )
            )

            start_frame = None

    if start_frame is not None:

        regions.append(
            (
                start_frame,
                len(active) - 1
            )
        )

    # Convert frames to timestamps
    converted = []

    for start_frame, end_frame in regions:

        start_time, _ = get_frame_times(
            start_frame,
            audio_duration
        )

        _, end_time = get_frame_times(
            end_frame,
            audio_duration
        )

        converted.append(
            {
                "start": start_time,
                "end": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )

    return converted


# ============================================================
# MERGE SMALL GAPS
# ============================================================

def merge_small_gaps(
    regions
):

    if not regions:

        return []

    merged = [
        dict(regions[0])
    ]

    for current in regions[1:]:

        previous = merged[-1]

        gap = (
            current["start"]
            - previous["end"]
        )

        if gap <= MAX_GAP:

            previous["end"] = max(
                previous["end"],
                current["end"]
            )

            previous["end_frame"] = max(
                previous["end_frame"],
                current["end_frame"]
            )

        else:

            merged.append(
                dict(current)
            )

    return merged


# ============================================================
# BUILD EVENTS
# ============================================================

def build_events(
    scores,
    audio_duration
):

    events = []

    event_counter = 1

    for target in TARGET_TO_YAMNET:

        frame_scores = get_target_scores(
            scores,
            target
        )

        regions = find_active_regions(
            frame_scores,
            audio_duration
        )

        regions = merge_small_gaps(
            regions
        )

        for region in regions:

            duration = (
                region["end"]
                - region["start"]
            )

            if duration < MIN_EVENT_DURATION:

                continue

            source_frames = list(
                range(
                    region["start_frame"],
                    region["end_frame"] + 1
                )
            )

            confidence = float(
                np.max(
                    frame_scores[
                        region["start_frame"]:
                        region["end_frame"] + 1
                    ]
                )
            )

            events.append(
                {
                    "event_id":
                        f"event_{event_counter:03d}",

                    "label":
                        target,

                    "start":
                        round(
                            region["start"],
                            2
                        ),

                    "end":
                        round(
                            region["end"],
                            2
                        ),

                    "duration":
                        round(
                            duration,
                            2
                        ),

                    "confidence":
                        round(
                            confidence,
                            3
                        ),

                    "source_frames":
                        source_frames,
                }
            )

            event_counter += 1

    # Sort chronologically
    events.sort(
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    # Reassign IDs after sorting
    for i, event in enumerate(
        events,
        start=1
    ):

        event["event_id"] = (
            f"event_{i:03d}"
        )

    return events


# ============================================================
# FRAME LEVEL CONTEXT
# ============================================================

def build_frame_context(
    scores,
    audio_duration
):

    frames = []

    for frame_index in range(
        scores.shape[0]
    ):

        start, end = get_frame_times(
            frame_index,
            audio_duration
        )

        labels = []

        for target in TARGET_TO_YAMNET:

            target_scores = (
                get_target_scores(
                    scores,
                    target
                )
            )

            confidence = float(
                target_scores[
                    frame_index
                ]
            )

            if confidence >= THRESHOLD:

                labels.append(
                    {
                        "label":
                            target,

                        "confidence":
                            round(
                                confidence,
                                3
                            ),
                    }
                )

        if labels:

            frames.append(
                {
                    "frame_index":
                        frame_index,

                    "start":
                        round(
                            start,
                            2
                        ),

                    "end":
                        round(
                            end,
                            2
                        ),

                    "labels":
                        labels,
                }
            )

    return frames


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    audio_path
):

    audio_path = Path(
        audio_path
    )

    if not audio_path.exists():

        print(
            f"ERROR: Audio file not found: "
            f"{audio_path}"
        )

        return

    audio, duration = load_audio(
        audio_path
    )

    scores = run_yamnet(
        audio
    )

    print(
        "\nBuilding frame level context..."
    )

    frames = build_frame_context(
        scores,
        duration
    )

    events = build_events(
        scores,
        duration
    )

    context = {

        "audio": {

            "file":
                audio_path.name,

            "duration":
                round(
                    duration,
                    2
                ),

            "sample_rate":
                TARGET_SR,
        },

        "model": {

            "name":
                "YAMNet",

            "threshold":
                THRESHOLD,

            "min_event_duration":
                MIN_EVENT_DURATION,

            "max_gap":
                MAX_GAP,

            "frame_duration":
                FRAME_DURATION,

            "frame_hop":
                FRAME_HOP,
        },

        "events":
            events,

        "frames":
            frames,
    }

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path(
        "data/context"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{audio_path.stem}_context.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            context,
            f,
            indent=2
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\nFrame activity:"
    )

    for frame in frames:

        print(
            f"  frame "
            f"{frame['frame_index']:02d} "
            f"{frame['start']:.2f}-"
            f"{frame['end']:.2f}: "
            + ", ".join(
                f"{x['label']}="
                f"{x['confidence']:.3f}"
                for x in frame["labels"]
            )
        )

    print(
        "\nDetected events:"
    )

    for event in events:

        print(
            f"  {event['event_id']} "
            f"{event['label']} "
            f"{event['start']:.2f}-"
            f"{event['end']:.2f} "
            f"duration="
            f"{event['duration']:.2f}s "
            f"confidence="
            f"{event['confidence']:.3f}"
        )

    print(
        f"\nTotal events: "
        f"{len(events)}"
    )

    print(
        f"Saved context: "
        f"{output_file}"
    )

    print(
        "\nContext generation complete."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python scripts/build_context.py "
            "<audio_file>"
        )

        sys.exit(1)

    build_context(
        sys.argv[1]
    )