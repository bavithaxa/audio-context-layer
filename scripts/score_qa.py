import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_FILE = Path("results/qa_predictions.json")
RESULTS_DIR = Path("results")

SUMMARY_FILE = RESULTS_DIR / "qa_metrics.json"
DETAIL_FILE = RESULTS_DIR / "qa_scored_predictions.json"


# ============================================================
# LABEL ALIASES
# ============================================================

ALIASES = {
    "car_horn": ["car_horn", "car horn", "horn"],
    "chirping_birds": [
        "chirping_birds",
        "chirping birds",
        "bird chirping",
        "chirping",
        "birds",
    ],
    "clock_tick": [
        "clock_tick",
        "clock tick",
        "ticking clock",
        "clock",
        "tick",
    ],
    "cow": ["cow", "cows"],
    "crow": ["crow", "crows"],
    "dog": [
        "dog",
        "dogs",
        "barking dog",
        "bark",
    ],
    "door_wood_knock": [
        "door_wood_knock",
        "door wood knock",
        "door knock",
        "knocking",
        "knock",
    ],
    "engine": ["engine", "engines", "motor"],
    "footsteps": [
        "footsteps",
        "footstep",
        "walking",
        "walk",
    ],
    "hen": ["hen", "hens"],
    "keyboard_typing": [
        "keyboard_typing",
        "keyboard typing",
        "typing",
        "keyboard",
    ],
    "laughing": [
        "laughing",
        "laughter",
        "laugh",
    ],
    "mouse_click": [
        "mouse_click",
        "mouse click",
        "clicking mouse",
    ],
    "rain": ["rain", "raining"],
    "rooster": ["rooster", "roosters"],
    "sheep": ["sheep"],
    "siren": ["siren", "sirens"],
    "thunderstorm": [
        "thunderstorm",
        "thunder",
    ],
    "water_drops": [
        "water_drops",
        "water drops",
        "water drop",
        "dripping water",
        "dripping",
        "drips",
    ],
    "wind": ["wind", "wind noise"],
}


# ============================================================
# JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("_", " ")

    text = re.sub(
        r"[^a-z0-9\s.]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LABEL EXTRACTION
# ============================================================

def extract_labels(text):

    text = normalize(text)

    found = set()

    aliases = []

    for label, values in ALIASES.items():

        for value in values:

            aliases.append(
                (
                    len(value),
                    label,
                    normalize(value)
                )
            )

    aliases.sort(reverse=True)

    for _, label, alias in aliases:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            text
        ):

            found.add(label)

    return found


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_number(text):

    if text is None:
        return None

    text = normalize(text)

    match = re.search(
        r"\b\d+(?:\.\d+)?\b",
        text
    )

    if match:

        value = float(match.group())

        if value.is_integer():
            return int(value)

        return value

    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    for word, value in words.items():

        if re.search(
            rf"\b{word}\b",
            text
        ):

            return value

    return None


# ============================================================
# YES / NO
# ============================================================

def extract_yes_no(text):

    text = normalize(text)

    if re.match(
        r"^(yes|yeah|yep|true)\b",
        text
    ):

        return True

    if re.match(
        r"^(no|nope|false)\b",
        text
    ):

        return False

    return None


# ============================================================
# COUNTING
# ============================================================

def score_counting(gt, pred):

    gt_number = extract_number(gt)
    pred_number = extract_number(pred)

    if (
        gt_number is None
        or pred_number is None
    ):

        return False

    return gt_number == pred_number


# ============================================================
# YES / NO
# ============================================================

def score_yes_no(gt, pred):

    gt_value = extract_yes_no(gt)
    pred_value = extract_yes_no(pred)

    if (
        gt_value is None
        or pred_value is None
    ):

        return False

    return gt_value == pred_value


# ============================================================
# APPARENT
# ============================================================

def score_apparent(gt, pred):

    gt_labels = extract_labels(gt)
    pred_labels = extract_labels(pred)

    tp = len(
        gt_labels & pred_labels
    )

    fp = len(
        pred_labels - gt_labels
    )

    fn = len(
        gt_labels - pred_labels
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": (
            gt_labels == pred_labels
        ),
        "gt_labels": sorted(gt_labels),
        "pred_labels": sorted(pred_labels),
    }


# ============================================================
# TEMPORAL
# ============================================================

def score_temporal(question, gt, pred):

    gt_labels = extract_labels(gt)
    pred_labels = extract_labels(pred)

    if len(gt_labels) == 1:

        expected = next(
            iter(gt_labels)
        )

        return expected in pred_labels

    return normalize(gt) in normalize(pred)


# ============================================================
# COMPARISON
# ============================================================

def comparison_relation(
    question,
    answer
):

    answer_text = normalize(answer)
    question_labels = extract_labels(question)

    equal_phrases = [
        "same amount",
        "same duration",
        "same length",
        "equally",
        "equal",
        "the same",
        "same",
        "tie",
        "tied",
    ]

    for phrase in equal_phrases:

        if phrase in answer_text:

            return (
                "equal",
                None
            )

    answer_labels = (
        question_labels
        & extract_labels(answer)
    )

    if len(answer_labels) == 1:

        return (
            "label",
            next(iter(answer_labels))
        )

    return (
        "unknown",
        None
    )


def score_comparison(
    question,
    gt,
    pred
):

    gt_relation = comparison_relation(
        question,
        gt
    )

    pred_relation = comparison_relation(
        question,
        pred
    )

    gt_type, gt_label = gt_relation
    pred_type, pred_label = pred_relation

    if gt_type == "unknown":
        return False

    if gt_type == "equal":

        return pred_type == "equal"

    return (
        pred_type == "label"
        and pred_label == gt_label
    )


# ============================================================
# DURATION
# ============================================================

def score_duration(gt, pred):

    gt_labels = extract_labels(gt)
    pred_labels = extract_labels(pred)

    if gt_labels:

        if not (
            gt_labels & pred_labels
        ):

            return False

    gt_numbers = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        normalize(gt)
    )

    pred_numbers = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        normalize(pred)
    )

    if gt_numbers and pred_numbers:

        gt_value = float(
            gt_numbers[0]
        )

        pred_value = float(
            pred_numbers[0]
        )

        tolerance = max(
            0.5,
            gt_value * 0.15
        )

        return abs(
            gt_value - pred_value
        ) <= tolerance

    return False


# ============================================================
# CAUSAL REASONING
# ============================================================

ENVIRONMENT_KEYWORDS = {

    "traffic roadside": [
        "traffic",
        "road",
        "roadside",
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "urban traffic",
    ],

    "rainy stormy": [
        "rain",
        "rainy",
        "storm",
        "stormy",
        "thunderstorm",
        "weather",
    ],

    "farm rural": [
        "farm",
        "rural",
        "countryside",
        "agricultural",
    ],

    "outdoor residential": [
        "residential",
        "outdoor",
        "neighborhood",
        "suburban",
    ],

    "indoor room": [
        "indoor",
        "room",
        "office",
        "house",
        "home",
    ],
}


def detect_environment(text):

    text = normalize(text)

    matches = set()

    for environment, keywords in (
        ENVIRONMENT_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in text:

                matches.add(
                    environment
                )

                break

    return matches


def score_causal(gt, pred):

    gt_environment = detect_environment(gt)
    pred_environment = detect_environment(pred)

    correct = bool(
        gt_environment
        & pred_environment
    )

    return {
        "environment_correct": correct,
        "gt_environment": sorted(
            gt_environment
        ),
        "pred_environment": sorted(
            pred_environment
        ),
        "answered": bool(
            normalize(pred)
        ),
    }


# ============================================================
# SCORE ONE RESULT
# ============================================================

def score_one(result):

    qtype = result.get(
        "question_type",
        ""
    )

    question = result.get(
        "question",
        ""
    )

    gt = result.get(
    "ground_truth",
    result.get("answer", "")
    )

    pred = result.get(
        "prediction",
        ""
    )

    if result.get("error"):

        return {
            "correct": False,
            "score": 0.0,
            "scorable": True,
        }

    if qtype == "counting":

        correct = score_counting(
            gt,
            pred
        )

        return {
            "correct": correct,
            "score": float(correct),
            "scorable": True,
        }

    if qtype == "yes_no":

        correct = score_yes_no(
            gt,
            pred
        )

        return {
            "correct": correct,
            "score": float(correct),
            "scorable": True,
        }

    if qtype == "apparent":

        return score_apparent(
            gt,
            pred
        ) | {
            "scorable": True
        }

    if qtype == "temporal":

        correct = score_temporal(
            question,
            gt,
            pred
        )

        return {
            "correct": correct,
            "score": float(correct),
            "scorable": True,
        }

    if qtype == "comparison":

        correct = score_comparison(
            question,
            gt,
            pred
        )

        return {
            "correct": correct,
            "score": float(correct),
            "scorable": True,
        }

    if qtype == "duration":

        correct = score_duration(
            gt,
            pred
        )

        return {
            "correct": correct,
            "score": float(correct),
            "scorable": True,
        }

    if qtype == "causal_reasoning":

        causal = score_causal(
            gt,
            pred
        )

        return {
            "correct": causal[
                "environment_correct"
            ],
            "score": float(
                causal[
                    "environment_correct"
                ]
            ),
            "scorable": True,
            "causal": causal,
        }

    return {
        "correct": False,
        "score": 0.0,
        "scorable": False,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AUDIO QA SCORING")
    print("=" * 65)

    predictions = load_json(
        PREDICTIONS_FILE
    )

    print(
        f"Loaded predictions: "
        f"{len(predictions)}"
    )

    scored_results = []

    for result in predictions:

        scored = dict(result)

        scored["score"] = score_one(
            result
        )

        scored_results.append(
            scored
        )

    # --------------------------------------------------------
    # Group results
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for result in scored_results:

        grouped[
            result["question_type"]
        ].append(result)

    metrics = {}

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    for qtype, items in grouped.items():

        if qtype == "apparent":

            precisions = [
                x["score"]["precision"]
                for x in items
            ]

            recalls = [
                x["score"]["recall"]
                for x in items
            ]

            f1s = [
                x["score"]["f1"]
                for x in items
            ]

            metrics[qtype] = {

                "count": len(items),

                "precision":
                    sum(precisions)
                    / len(precisions),

                "recall":
                    sum(recalls)
                    / len(recalls),

                "f1":
                    sum(f1s)
                    / len(f1s),
            }

        elif qtype == "causal_reasoning":

            correct = sum(
                1
                for x in items
                if x["score"].get(
                    "correct",
                    False
                )
            )

            metrics[qtype] = {

                "count": len(items),

                "correct": correct,

                "accuracy":
                    correct / len(items)
                    if items
                    else 0.0,

                "environment_correct":
                    correct,

                "environment_accuracy":
                    correct / len(items)
                    if items
                    else 0.0,
            }

        else:

            correct = sum(
                1
                for x in items
                if x["score"].get(
                    "correct",
                    False
                )
            )

            metrics[qtype] = {

                "count": len(items),

                "correct": correct,

                "accuracy":
                    correct / len(items)
                    if items
                    else 0.0,
            }

    # --------------------------------------------------------
    # Overall objective accuracy
    # --------------------------------------------------------

    objective = [
        x
        for x in scored_results
        if x["question_type"]
        != "causal_reasoning"
    ]

    objective_correct = sum(
        1
        for x in objective
        if x["score"].get(
            "correct",
            False
        )
    )

    objective_accuracy = (
        objective_correct
        / len(objective)
        if objective
        else 0.0
    )

    metrics[
        "overall_objective"
    ] = {

        "count": len(objective),

        "correct": objective_correct,

        "accuracy":
            objective_accuracy,
    }

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_json(
        SUMMARY_FILE,
        metrics
    )

    save_json(
        DETAIL_FILE,
        scored_results
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)

    for qtype in sorted(
        metrics.keys()
    ):

        data = metrics[qtype]

        print()
        print(qtype)

        # Apparent
        if "precision" in data:

            print(
                f"  Precision: "
                f"{data['precision']:.4f}"
            )

            print(
                f"  Recall: "
                f"{data['recall']:.4f}"
            )

            print(
                f"  F1: "
                f"{data['f1']:.4f}"
            )

        # Causal
        elif (
            qtype == "causal_reasoning"
            and "environment_accuracy"
            in data
        ):

            print(
                f"  Environment accuracy: "
                f"{data['environment_accuracy']:.4f}"
            )

            print(
                f"  Correct: "
                f"{data['environment_correct']}/"
                f"{data['count']}"
            )

        # Everything else
        else:

            print(
                f"  Accuracy: "
                f"{data['accuracy']:.4f}"
            )

            print(
                f"  Correct: "
                f"{data['correct']}/"
                f"{data['count']}"
            )

    print()
    print(
        "Overall objective accuracy: "
        f"{objective_accuracy:.4f}"
    )

    print()
    print(
        f"Metrics saved to: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Detailed scores saved to: "
        f"{DETAIL_FILE}"
    )


if __name__ == "__main__":
    main()