import json
import random
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_DIR = BASE_DIR / "data" / "manifests"
QA_DIR = BASE_DIR / "data" / "qa"

QA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
QUESTIONS_PER_SCENE = 8

random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_events(events):
    return sorted(events, key=lambda x: x["start"])


def event_description(event):
    return event["label"].replace("_", " ")


def format_time(seconds):
    return round(seconds, 2)


def choose_event(events, exclude_labels=None):
    """
    Choose one event randomly.
    """
    exclude_labels = exclude_labels or []

    candidates = [
        event
        for event in events
        if event["label"] not in exclude_labels
    ]

    if not candidates:
        candidates = events

    return random.choice(candidates)


def choose_two_different_events(events):
    """
    Choose two events with different labels when possible.
    """

    labels = list(
        set(event["label"] for event in events)
    )

    if len(labels) >= 2:

        label_a, label_b = random.sample(labels, 2)

        event_a = random.choice([
            event
            for event in events
            if event["label"] == label_a
        ])

        event_b = random.choice([
            event
            for event in events
            if event["label"] == label_b
        ])

        return event_a, event_b

    return events[0], events[-1]


def make_qa(
    qa_id,
    scene_id,
    split,
    question_type,
    question,
    answer,
    evidence_event_ids=None
):
    """
    Create one QA record.
    """

    return {
        "qa_id": qa_id,
        "scene_id": scene_id,
        "split": split,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "evidence_event_ids": evidence_event_ids or []
    }


# ============================================================
# QUESTION TYPE 1: APPARENT
# ============================================================

def generate_apparent_question(scene, qa_id):

    events = scene["events"]

    labels = sorted(
        set(event["label"] for event in events)
    )

    readable_labels = [
        label.replace("_", " ")
        for label in labels
    ]

    answer = ", ".join(readable_labels)

    evidence = [
        event["event_id"]
        for event in events
    ]

    question = (
        "What distinct sounds are present "
        "in the audio scene?"
    )

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "apparent",
        question,
        answer,
        evidence
    )


# ============================================================
# QUESTION TYPE 2: COUNTING
# ============================================================

def generate_counting_question(scene, qa_id):

    events = scene["events"]

    label_counts = Counter(
        event["label"]
        for event in events
    )

    repeated_labels = [
        label
        for label, count in label_counts.items()
        if count >= 2
    ]

    if repeated_labels:

        label = random.choice(repeated_labels)

    else:

        label = random.choice(
            list(label_counts.keys())
        )

    count = label_counts[label]

    evidence = [
        event["event_id"]
        for event in events
        if event["label"] == label
    ]

    readable_label = label.replace(
        "_",
        " "
    )

    question = (
        f"How many separate {readable_label} "
        f"events occur in the scene?"
    )

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "counting",
        question,
        str(count),
        evidence
    )


# ============================================================
# QUESTION TYPE 3: TEMPORAL
# ============================================================

def generate_temporal_question(scene, qa_id):

    events = sort_events(
        scene["events"]
    )

    event_a, event_b = choose_two_different_events(
        events
    )

    if event_a["start"] <= event_b["start"]:

        earlier = event_a
        later = event_b

    else:

        earlier = event_b
        later = event_a

    earlier_label = event_description(
        earlier
    )

    later_label = event_description(
        later
    )

    question = (
        f"Which occurred first, the "
        f"{earlier_label} or the {later_label}?"
    )

    answer = earlier_label

    evidence = [
        earlier["event_id"],
        later["event_id"]
    ]

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "temporal",
        question,
        answer,
        evidence
    )


# ============================================================
# QUESTION TYPE 4: YES / NO
# ============================================================

def generate_yes_no_question(scene, qa_id):

    events = scene["events"]

    existing_labels = list(
        set(
            event["label"]
            for event in events
        )
    )

    # --------------------------------------------
    # 70% chance: ask about an event that exists
    # --------------------------------------------

    if random.random() < 0.7:

        label = random.choice(
            existing_labels
        )

        answer = "yes"

        evidence = [
            event["event_id"]
            for event in events
            if event["label"] == label
        ]

    # --------------------------------------------
    # 30% chance: ask about an absent event
    # --------------------------------------------

    else:

        all_labels = set()

        for manifest_path in MANIFEST_DIR.glob(
            "scene_*.json"
        ):

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as f:

                other_scene = json.load(f)

            for event in other_scene["events"]:

                all_labels.add(
                    event["label"]
                )

        absent_labels = list(
            all_labels - set(existing_labels)
        )

        if absent_labels:

            label = random.choice(
                absent_labels
            )

            answer = "no"

            evidence = []

        else:

            label = random.choice(
                existing_labels
            )

            answer = "yes"

            evidence = [
                event["event_id"]
                for event in events
                if event["label"] == label
            ]

    readable_label = label.replace(
        "_",
        " "
    )

    question = (
        f"Is there a {readable_label} "
        f"sound in the scene?"
    )

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "yes_no",
        question,
        answer,
        evidence
    )


# ============================================================
# QUESTION TYPE 5: COMPARISON
# ============================================================

def generate_comparison_question(scene, qa_id):

    events = scene["events"]

    event_a, event_b = choose_two_different_events(
        events
    )

    duration_a = (
        event_a["end"]
        -
        event_a["start"]
    )

    duration_b = (
        event_b["end"]
        -
        event_b["start"]
    )

    label_a = event_description(
        event_a
    )

    label_b = event_description(
        event_b
    )

    if duration_a > duration_b:

        answer = label_a

    elif duration_b > duration_a:

        answer = label_b

    else:

        answer = (
            "They lasted the same amount of time."
        )

    question = (
        f"Which lasted longer, the "
        f"{label_a} or the {label_b}?"
    )

    evidence = [
        event_a["event_id"],
        event_b["event_id"]
    ]

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "comparison",
        question,
        answer,
        evidence
    )


# ============================================================
# QUESTION TYPE 6: DURATION
# ============================================================

def generate_duration_question(scene, qa_id):

    events = scene["events"]

    event = random.choice(events)

    duration = (
        event["end"]
        -
        event["start"]
    )

    label = event_description(
        event
    )

    question = (
        f"Approximately how long does the "
        f"{label} event last?"
    )

    answer = (
        f"{format_time(duration)} seconds"
    )

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "duration",
        question,
        answer,
        [event["event_id"]]
    )


# ============================================================
# QUESTION TYPE 7: CAUSAL / REASONING
# ============================================================

def generate_reasoning_question(scene, qa_id):

    hidden = scene.get(
        "hidden_reasoning",
        {}
    )

    environment = hidden.get(
        "environment",
        scene.get(
            "scene_type",
            "unknown"
        )
    )

    reference_basis = hidden.get(
        "reference_basis",
        ""
    )

    readable_environment = (
        environment.replace(
            "_",
            " "
        )
    )

    # Correct article: a / an
    first_letter = (
        readable_environment[0].lower()
        if readable_environment
        else ""
    )

    article = (
        "an"
        if first_letter in "aeiou"
        else "a"
    )

    answer = (
        f"{article.capitalize()} "
        f"{readable_environment} environment "
        f"is suggested. "
        f"{reference_basis}"
    )

    question = (
        "What type of environment is suggested "
        "by the sounds, and what evidence supports this?"
    )

    evidence = [
        event["event_id"]
        for event in scene["events"]
    ]

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "causal_reasoning",
        question,
        answer,
        evidence
    )


# ============================================================
# QUESTION TYPE 8: LAST EVENT
# ============================================================

def generate_event_order_question(scene, qa_id):

    events = sort_events(
        scene["events"]
    )

    last_event = events[-1]

    last_label = event_description(
        last_event
    )

    question = (
        "What sound occurs last in the scene?"
    )

    answer = last_label

    evidence = [
        last_event["event_id"]
    ]

    return make_qa(
        qa_id,
        scene["scene_id"],
        scene["split"],
        "temporal",
        question,
        answer,
        evidence
    )


# ============================================================
# GENERATE QUESTIONS FOR ONE SCENE
# ============================================================

def generate_questions(scene):

    scene_id = scene["scene_id"]

    generators = [
        generate_apparent_question,
        generate_counting_question,
        generate_temporal_question,
        generate_yes_no_question,
        generate_comparison_question,
        generate_duration_question,
        generate_reasoning_question,
        generate_event_order_question
    ]

    questions = []

    for index, generator in enumerate(
        generators,
        start=1
    ):

        qa_id = (
            f"{scene_id}_q{index:02d}"
        )

        qa = generator(
            scene,
            qa_id
        )

        questions.append(qa)

    return questions


# ============================================================
# MAIN
# ============================================================

def main():

    manifest_paths = sorted(
        MANIFEST_DIR.glob(
            "scene_*.json"
        )
    )

    all_questions = []

    print("=" * 60)
    print("AUDIO CONTEXT LAYER - QA GENERATION")
    print("=" * 60)

    print(
        f"\nFound {len(manifest_paths)} "
        f"scene manifests."
    )

    # --------------------------------------------------------
    # Generate QA for every scene
    # --------------------------------------------------------

    for manifest_path in manifest_paths:

        scene = load_manifest(
            manifest_path
        )

        questions = generate_questions(
            scene
        )

        all_questions.extend(
            questions
        )

        # Save individual scene QA
        output_path = (
            QA_DIR
            /
            f"{scene['scene_id']}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                questions,
                f,
                indent=2,
                ensure_ascii=False
            )

    # --------------------------------------------------------
    # Save combined QA dataset
    # --------------------------------------------------------

    combined_path = (
        QA_DIR
        /
        "qa_all.json"
    )

    with open(
        combined_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_questions,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    type_counts = Counter(
        qa["question_type"]
        for qa in all_questions
    )

    split_counts = Counter(
        qa["split"]
        for qa in all_questions
    )

    print("\n--- QA DATASET ---")

    print(
        f"Total QA pairs: "
        f"{len(all_questions)}"
    )

    print(
        f"Questions per scene: "
        f"{QUESTIONS_PER_SCENE}"
    )

    print("\n--- SPLITS ---")

    for split in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"{split:>12}: "
            f"{split_counts[split]} QA pairs"
        )

    print("\n--- QUESTION TYPES ---")

    for question_type, count in sorted(
        type_counts.items()
    ):

        print(
            f"{question_type:>20}: "
            f"{count}"
        )

    print("\n--- OUTPUT ---")

    print(
        f"Individual QA files: "
        f"{len(manifest_paths)}"
    )

    print(
        f"Combined dataset: "
        f"{combined_path}"
    )

    print("\n" + "=" * 60)
    print("✅ QA GENERATION COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()