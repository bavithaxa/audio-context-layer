import json
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCENE_DIR = BASE_DIR / "data" / "scenes"
MANIFEST_DIR = BASE_DIR / "data" / "manifests"


# ============================================================
# HELPERS
# ============================================================

def overlaps(event_a, event_b):
    """
    Check whether two events overlap in time.
    """
    return (
        event_a["start"] < event_b["end"]
        and event_b["start"] < event_a["end"]
    )


# ============================================================
# LOAD MANIFESTS
# ============================================================

manifests = sorted(MANIFEST_DIR.glob("scene_*.json"))

print("=" * 60)
print("AUDIO CONTEXT LAYER - DATASET SANITY CHECK")
print("=" * 60)

print(f"\nFound manifests: {len(manifests)}")

if len(manifests) != 150:
    print("WARNING: Expected 150 scenes.")


# ============================================================
# COUNTERS
# ============================================================

split_counts = Counter()
scene_type_counts = Counter()

durations = []
events_per_scene = []

scenes_with_overlap = 0
scenes_with_repetition = 0

missing_audio = []
invalid_timestamps = []

source_files_by_split = defaultdict(set)


# ============================================================
# INSPECT EACH SCENE
# ============================================================

for manifest_path in manifests:

    with open(manifest_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    split = scene["split"]
    duration = scene["duration"]
    events = scene["events"]

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    split_counts[split] += 1

    # --------------------------------------------------------
    # Scene type
    # --------------------------------------------------------

    scene_type = scene.get("scene_type", "unknown")
    scene_type_counts[scene_type] += 1

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    durations.append(duration)

    # --------------------------------------------------------
    # Number of events
    # --------------------------------------------------------

    events_per_scene.append(len(events))

    # --------------------------------------------------------
    # Check audio file
    # --------------------------------------------------------

    audio_path = SCENE_DIR / f"{scene_id}.wav"

    if not audio_path.exists():
        missing_audio.append(scene_id)

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    for event in events:

        start = event["start"]
        end = event["end"]

        if start < 0 or end <= start or end > duration:

            invalid_timestamps.append(
                (scene_id, event["event_id"])
            )

        # Track source files for split leakage check
        source_file = event.get("source_file")

        if source_file:
            source_files_by_split[split].add(source_file)

    # --------------------------------------------------------
    # Repeated event labels
    # --------------------------------------------------------

    labels = [
        event["label"]
        for event in events
    ]

    label_counts = Counter(labels)

    if any(count > 1 for count in label_counts.values()):
        scenes_with_repetition += 1

    # --------------------------------------------------------
    # Overlap detection
    # --------------------------------------------------------

    has_overlap = False

    for i in range(len(events)):

        for j in range(i + 1, len(events)):

            if overlaps(events[i], events[j]):
                has_overlap = True
                break

        if has_overlap:
            break

    if has_overlap:
        scenes_with_overlap += 1


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n--- SPLITS ---")

for split in ["train", "validation", "test"]:

    print(
        f"{split:>12}: "
        f"{split_counts[split]} scenes"
    )


print("\n--- SCENE TYPES ---")

for scene_type, count in sorted(scene_type_counts.items()):

    print(
        f"{scene_type:>25}: "
        f"{count}"
    )


print("\n--- AUDIO ---")

print(
    f"Total scenes: "
    f"{len(manifests)}"
)

print(
    f"Missing audio: "
    f"{len(missing_audio)}"
)


print("\n--- DURATIONS ---")

if durations:

    print(
        f"Minimum: "
        f"{min(durations):.2f}s"
    )

    print(
        f"Maximum: "
        f"{max(durations):.2f}s"
    )

    print(
        f"Average: "
        f"{sum(durations) / len(durations):.2f}s"
    )


print("\n--- EVENTS ---")

if events_per_scene:

    print(
        f"Minimum events/scene: "
        f"{min(events_per_scene)}"
    )

    print(
        f"Maximum events/scene: "
        f"{max(events_per_scene)}"
    )

    print(
        f"Average events/scene: "
        f"{sum(events_per_scene) / len(events_per_scene):.2f}"
    )


print("\n--- REPETITION ---")

print(
    f"Scenes containing repeated event labels: "
    f"{scenes_with_repetition}/{len(manifests)}"
)


print("\n--- OVERLAP ---")

print(
    f"Scenes containing overlapping events: "
    f"{scenes_with_overlap}/{len(manifests)}"
)


print("\n--- TIMESTAMPS ---")

print(
    f"Invalid timestamp events: "
    f"{len(invalid_timestamps)}"
)


# ============================================================
# CROSS SPLIT SOURCE LEAKAGE CHECK
# ============================================================

print("\n--- CROSS SPLIT SOURCE CHECK ---")

splits = [
    "train",
    "validation",
    "test"
]

cross_split_sources = []

for i in range(len(splits)):

    for j in range(i + 1, len(splits)):

        split_a = splits[i]
        split_b = splits[j]

        shared_sources = (
            source_files_by_split[split_a]
            &
            source_files_by_split[split_b]
        )

        if shared_sources:

            cross_split_sources.append(
                (
                    split_a,
                    split_b,
                    shared_sources
                )
            )

        print(
            f"{split_a} vs {split_b}: "
            f"{len(shared_sources)} shared source files"
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

problems = []


# Correct number of scenes

if len(manifests) != 150:

    problems.append(
        "Total number of scenes is not 150."
    )


# Correct splits

if split_counts["train"] != 100:

    problems.append(
        "Train split is not 100 scenes."
    )


if split_counts["validation"] != 25:

    problems.append(
        "Validation split is not 25 scenes."
    )


if split_counts["test"] != 25:

    problems.append(
        "Test split is not 25 scenes."
    )


# Audio files

if missing_audio:

    problems.append(
        "Some scenes are missing audio files."
    )


# Timestamp validation

if invalid_timestamps:

    problems.append(
        "Some events have invalid timestamps."
    )


# Source leakage

if cross_split_sources:

    problems.append(
        "Some source recordings appear across multiple splits."
    )


# ============================================================
# FINAL RESULT
# ============================================================

if problems:

    print("\n⚠️ Problems found:\n")

    for problem in problems:

        print(
            f"- {problem}"
        )

else:

    print(
        "\n✅ Dataset sanity check passed!"
    )

    print(
        "The dataset is ready for QA generation."
    )