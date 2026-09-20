import csv
import json
import random
from pathlib import Path

from pydub import AudioSegment


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "ESC-50-master" / "meta" / "esc50.csv"
AUDIO_DIR = PROJECT_ROOT / "ESC-50-master" / "audio"

SCENES_DIR = PROJECT_ROOT / "data" / "scenes"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"

SCENES_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)


# ============================================================
# 3. DATASET SPLITS
# ============================================================

SPLITS = {
    "train": {
        "count": 100,
        "folds": {1, 2, 3}
    },
    "validation": {
        "count": 25,
        "folds": {4}
    },
    "test": {
        "count": 25,
        "folds": {5}
    }
}


# ============================================================
# 4. SCENE THEMES
# ============================================================

SCENE_TEMPLATES = {

    "outdoor_residential": {
        "background": [
            "chirping_birds",
            "wind"
        ],
        "foreground": [
            "dog",
            "footsteps",
            "car_horn",
            "crow"
        ],
        "reasoning": (
            "Birds and wind provide evidence consistent with an "
            "outdoor environment. Dogs or footsteps can provide "
            "additional evidence of a residential outdoor setting."
        )
    },

    "rainy_stormy": {
        "background": [
            "rain",
            "wind"
        ],
        "foreground": [
            "thunderstorm",
            "water_drops",
            "car_horn",
            "footsteps"
        ],
        "reasoning": (
            "Rain and wind provide evidence of outdoor weather, "
            "while thunderstorm sounds are consistent with stormy "
            "conditions."
        )
    },

    "indoor_room": {
        "background": [
            "clock_tick"
        ],
        "foreground": [
            "keyboard_typing",
            "mouse_click",
            "door_wood_knock",
            "footsteps",
            "laughing"
        ],
        "reasoning": (
            "Clock ticking, keyboard typing and mouse clicks are "
            "sounds consistent with an indoor room."
        )
    },

    "traffic_roadside": {
        "background": [
            "engine"
        ],
        "foreground": [
            "car_horn",
            "siren",
            "footsteps",
            "engine"
        ],
        "reasoning": (
            "Engine sounds, car horns and sirens provide evidence "
            "consistent with a road or traffic environment."
        )
    },

    "farm_rural": {
        "background": [
            "wind"
        ],
        "foreground": [
            "cow",
            "sheep",
            "rooster",
            "hen",
            "crow"
        ],
        "reasoning": (
            "Animal sounds such as cows, sheep, hens and roosters "
            "provide evidence consistent with a farm or rural "
            "environment."
        )
    }
}


# ============================================================
# 5. READ ESC 50
# ============================================================

with open(CSV_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)


# ============================================================
# 6. GROUP CLIPS BY SPLIT AND CATEGORY
# ============================================================

clips = {
    "train": {},
    "validation": {},
    "test": {}
}


def split_for_fold(fold):
    if fold in {1, 2, 3}:
        return "train"
    elif fold == 4:
        return "validation"
    elif fold == 5:
        return "test"


for row in rows:

    category = row["category"]
    fold = int(row["fold"])

    split = split_for_fold(fold)

    if split is None:
        continue

    if category not in clips[split]:
        clips[split][category] = []

    clips[split][category].append(row)


# ============================================================
# 7. SELECT A SOURCE RECORDING
# ============================================================

def choose_clip(category, split, used_files):

    available = [
        row
        for row in clips[split].get(category, [])
        if row["filename"] not in used_files
    ]

    if not available:
        available = clips[split].get(category, [])

    if not available:
        raise ValueError(
            f"No '{category}' recordings available "
            f"for split '{split}'."
        )

    return random.choice(available)


# ============================================================
# 8. ADD ONE EVENT
# ============================================================

def add_event(
    scene,
    events,
    category,
    split,
    start_ms,
    role,
    used_files
):

    source = choose_clip(
        category,
        split,
        used_files
    )

    used_files.add(source["filename"])

    audio_path = AUDIO_DIR / source["filename"]

    clip = AudioSegment.from_wav(audio_path)

    # Reduce volume slightly so multiple overlapping sounds
    # do not become excessively loud.
    clip = clip - 5

    # Keep the event inside the scene.
    if start_ms + len(clip) > len(scene):

        start_ms = max(
            0,
            len(scene) - len(clip)
        )

    scene = scene.overlay(
        clip,
        position=start_ms
    )

    events.append({
        "event_id": f"e{len(events) + 1}",
        "label": category,
        "start": round(start_ms / 1000, 2),
        "end": round(
            (start_ms + len(clip)) / 1000,
            2
        ),
        "source_file": source["filename"],
        "role": role
    })

    return scene


# ============================================================
# 9. CREATE ONE SCENE
# ============================================================

def create_scene(scene_number, split, scene_type):

    template = SCENE_TEMPLATES[scene_type]

    # Random scene duration between 20 and 30 seconds.
    duration_seconds = random.choice(
        [20, 22, 25, 28, 30]
    )

    duration_ms = duration_seconds * 1000

    # Begin with complete silence.
    scene = AudioSegment.silent(
        duration=duration_ms
    )

    events = []
    used_files = set()

    # --------------------------------------------------------
    # Background sounds
    # --------------------------------------------------------

    for category in template["background"]:

        # Background sounds begin early,
        # creating persistent contextual overlap.
        start_ms = random.randint(
            0,
            min(5000, duration_ms - 5000)
        )

        scene = add_event(
            scene,
            events,
            category,
            split,
            start_ms,
            "background",
            used_files
        )

    # --------------------------------------------------------
    # Foreground sounds
    # --------------------------------------------------------

    foreground_count = random.randint(3, 4)

    for _ in range(foreground_count):

        category = random.choice(
            template["foreground"]
        )

        # Most foreground sounds occur somewhere
        # inside the scene, rather than only at boundaries.
        start_ms = random.randint(
            1000,
            max(1000, duration_ms - 5000)
        )

        scene = add_event(
            scene,
            events,
            category,
            split,
            start_ms,
            "foreground",
            used_files
        )

    # --------------------------------------------------------
    # Intentionally add one repeated event
    # --------------------------------------------------------

    repeat_category = random.choice(
        template["foreground"]
    )

    repeat_start_ms = random.randint(
        1000,
        max(1000, duration_ms - 5000)
    )

    scene = add_event(
        scene,
        events,
        repeat_category,
        split,
        repeat_start_ms,
        "foreground_repeat",
        used_files
    )

    # --------------------------------------------------------
    # Sort events by timestamp
    # --------------------------------------------------------

    events.sort(
        key=lambda event: event["start"]
    )

    # Reassign IDs after sorting.
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"e{index}"

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    scene_id = f"scene_{scene_number:03d}"

    audio_path = (
        SCENES_DIR /
        f"{scene_id}.wav"
    )

    scene.export(
        audio_path,
        format="wav"
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = {
        "scene_id": scene_id,
        "split": split,
        "scene_type": scene_type,
        "duration": duration_seconds,
        "seed": SEED,

        "events": events,

        "hidden_reasoning": {
            "environment": scene_type,
            "reference_basis": template["reasoning"]
        }
    }

    manifest_path = (
        MANIFESTS_DIR /
        f"{scene_id}.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    return scene_id


# ============================================================
# 10. GENERATE ALL 150 SCENES
# ============================================================

scene_number = 1

theme_names = list(
    SCENE_TEMPLATES.keys()
)

for split, split_info in SPLITS.items():

    count = split_info["count"]

    for _ in range(count):

        scene_type = random.choice(
            theme_names
        )

        scene_id = create_scene(
            scene_number,
            split,
            scene_type
        )

        print(
            f"Created {scene_id} | "
            f"{split} | "
            f"{scene_type}"
        )

        scene_number += 1


print("\n========================================")
print("150 synthetic scenes created successfully!")
print("========================================")

print("\nDataset split:")
print("Train      : 100 scenes")
print("Validation : 25 scenes")
print("Test       : 25 scenes")

print("\nEach scene:")
print("• 20–30 seconds")
print("• Multiple sound events")
print("• Overlapping events")
print("• Repeated events")
print("• Silence/background periods")
print("• Ground truth manifest")