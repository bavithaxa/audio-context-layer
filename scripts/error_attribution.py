import json
import re
import csv
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SCORED_FILE = Path("results/qa_scored_predictions.json")
MANIFEST_DIR = Path("data/manifests")
CONTEXT_DIR = Path("data/context")

OUTPUT_JSON = Path("results/error_attribution.json")
OUTPUT_CSV = Path("results/error_attribution.csv")


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

ALIASES = {
    "chirping birds": "chirping_birds",
    "chirping bird": "chirping_birds",
    "birds": "chirping_birds",
    "bird": "chirping_birds",

    "water drops": "water_drops",
    "water drop": "water_drops",

    "clock tick": "clock_tick",

    "door knock": "door_wood_knock",
    "door knocks": "door_wood_knock",
    "knocking": "door_wood_knock",

    "car horn": "car_horn",
    "car horns": "car_horn",

    "footstep": "footsteps",

    "keyboard typing": "keyboard_typing",
    "typing": "keyboard_typing",
    "keyboard": "keyboard_typing",

    "laughter": "laughing",

    "thunder": "thunderstorm",

    "siren": "siren",
    "engine": "engine",
    "dog": "dog",
    "cow": "cow",
    "sheep": "sheep",
    "hen": "hen",
    "rooster": "rooster",
    "crow": "crow",
    "rain": "rain",
    "wind": "wind",
    "mouse click": "mouse_click",
}


def normalize_label(label):

    if not label:
        return ""

    text = str(label).strip().lower()

    text = re.sub(
        r"[^a-z0-9_ ]+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if text in ALIASES:
        return ALIASES[text]

    return text.replace(" ", "_")


# ============================================================
# TEXT LABEL EXTRACTION
# ============================================================

def extract_labels(text):

    if not text:
        return set()

    text = str(text).lower()

    found = set()

    for alias, canonical in ALIASES.items():

        if alias in text:
            found.add(canonical)

    canonical_labels = set(
        ALIASES.values()
    )

    for label in canonical_labels:

        readable = label.replace(
            "_",
            " "
        )

        if (
            label in text
            or readable in text
        ):
            found.add(label)

    return found


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_manifest_events(manifest):

    events = manifest.get(
        "events",
        []
    )

    output = []

    for event in events:

        start = float(
            event.get("start", 0)
        )

        end = float(
            event.get("end", 0)
        )

        output.append({
            "label": normalize_label(
                event.get("label")
            ),
            "start": start,
            "end": end,
            "duration": max(
                0,
                end - start
            ),
        })

    return output


def extract_context_events(context):

    events = context.get(
        "events",
        []
    )

    output = []

    for event in events:

        start = float(
            event.get("start", 0)
        )

        end = float(
            event.get("end", 0)
        )

        output.append({
            "event_id": event.get(
                "event_id"
            ),
            "label": normalize_label(
                event.get("label")
            ),
            "start": start,
            "end": end,
            "duration": max(
                0,
                end - start
            ),
            "confidence": float(
                event.get(
                    "confidence",
                    0
                )
            ),
        })

    return output


# ============================================================
# TEMPORAL OVERLAP
# ============================================================

def overlap(
    a_start,
    a_end,
    b_start,
    b_end
):

    return max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )


def iou(gt, pred):

    intersection = overlap(
        gt["start"],
        gt["end"],
        pred["start"],
        pred["end"]
    )

    if intersection <= 0:
        return 0.0

    union = (
        gt["duration"]
        + pred["duration"]
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# EVENT MATCHING
# ============================================================

def match_events(
    gt_events,
    context_events,
    threshold=0.20
):

    matches = []
    used = set()

    for gt_index, gt in enumerate(
        gt_events
    ):

        best_index = None
        best_score = 0.0

        for pred_index, pred in enumerate(
            context_events
        ):

            if pred_index in used:
                continue

            if gt["label"] != pred["label"]:
                continue

            score = iou(
                gt,
                pred
            )

            if score > best_score:
                best_score = score
                best_index = pred_index

        if (
            best_index is not None
            and best_score >= threshold
        ):

            used.add(
                best_index
            )

            matches.append({
                "gt_index": gt_index,
                "pred_index": best_index,
                "iou": round(
                    best_score,
                    3
                ),
            })

    return matches


# ============================================================
# CONTEXT ANALYSIS
# ============================================================

def analyze_context(
    gt_events,
    context_events
):

    matches = match_events(
        gt_events,
        context_events
    )

    matched_gt = {
        m["gt_index"]
        for m in matches
    }

    matched_context = {
        m["pred_index"]
        for m in matches
    }

    missing = [
        gt_events[i]
        for i in range(
            len(gt_events)
        )
        if i not in matched_gt
    ]

    extra = [
        context_events[i]
        for i in range(
            len(context_events)
        )
        if i not in matched_context
    ]

    return {
        "matches": matches,
        "missing": missing,
        "extra": extra,
    }


# ============================================================
# QUESTION EVIDENCE
# ============================================================

def relevant_labels(
    question,
    answer
):

    labels = set()

    labels.update(
        extract_labels(question)
    )

    labels.update(
        extract_labels(answer)
    )

    return labels


def context_has_required_labels(
    question,
    answer,
    gt_events,
    context_events
):

    labels = relevant_labels(
        question,
        answer
    )

    if not labels:
        return None

    gt_labels = {
        e["label"]
        for e in gt_events
        if e["label"] in labels
    }

    context_labels = {
        e["label"]
        for e in context_events
        if e["label"] in labels
    }

    if not gt_labels:
        return None

    return gt_labels.issubset(
        context_labels
    )


# ============================================================
# TARGET LABEL EXTRACTION
# ============================================================

def question_target_labels(
    question,
    gt_events
):
    """
    Identify labels relevant to the question.

    This is especially important for counting and duration
    questions because the answer itself may only contain
    a number or duration.
    """

    question_labels = extract_labels(
        question
    )

    gt_labels = {
        e["label"]
        for e in gt_events
    }

    return question_labels.intersection(
        gt_labels
    )


# ============================================================
# COUNTING ANALYSIS
# ============================================================

def counting_error_reason(
    question,
    answer,
    gt_events,
    context_events
):

    labels = question_target_labels(
        question,
        gt_events
    )

    if not labels:
        return None

    for label in labels:

        gt_count = sum(
            e["label"] == label
            for e in gt_events
        )

        context_count = sum(
            e["label"] == label
            for e in context_events
        )

        if gt_count != context_count:

            return (
                f"Counting perception mismatch "
                f"for '{label}': "
                f"ground truth={gt_count}, "
                f"context={context_count}."
            )

    return None


# ============================================================
# DURATION ANALYSIS
# ============================================================

def duration_error_reason(
    question,
    answer,
    gt_events,
    context_events
):

    labels = question_target_labels(
        question,
        gt_events
    )

    if not labels:
        return None

    for label in labels:

        gt = [
            e for e in gt_events
            if e["label"] == label
        ]

        pred = [
            e for e in context_events
            if e["label"] == label
        ]

        if not gt:
            continue

        if not pred:

            return (
                f"Duration target '{label}' "
                "is missing from context."
            )

        gt_max = max(
            e["duration"]
            for e in gt
        )

        pred_max = max(
            e["duration"]
            for e in pred
        )

        # A difference greater than 1.5 seconds
        # indicates a meaningful boundary problem.
        if abs(
            gt_max - pred_max
        ) > 1.5:

            return (
                f"Duration boundary mismatch "
                f"for '{label}': "
                f"ground truth={gt_max:.2f}s, "
                f"context={pred_max:.2f}s."
            )

    return None


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def temporal_error_reason(
    question,
    answer,
    gt_events,
    context_events
):

    labels = extract_labels(
        question
    )

    for label in labels:

        gt_exists = any(
            e["label"] == label
            for e in gt_events
        )

        context_exists = any(
            e["label"] == label
            for e in context_events
        )

        if (
            gt_exists
            and not context_exists
        ):

            return (
                f"Temporal target '{label}' "
                "is missing from context."
            )

    return None


# ============================================================
# COMPARISON ANALYSIS
# ============================================================

def comparison_error_reason(
    question,
    answer,
    gt_events,
    context_events
):
    """
    Check whether the relative duration relationship
    between the two sounds changed between the manifest
    and generated context.
    """

    labels = question_target_labels(
        question,
        gt_events
    )

    if len(labels) < 2:
        return None

    labels = list(labels)[:2]

    gt_durations = {}
    context_durations = {}

    for label in labels:

        gt = [
            e["duration"]
            for e in gt_events
            if e["label"] == label
        ]

        context = [
            e["duration"]
            for e in context_events
            if e["label"] == label
        ]

        if not gt or not context:
            return (
                f"Comparison target '{label}' "
                "is missing from context."
            )

        gt_durations[label] = max(gt)
        context_durations[label] = max(context)

    a, b = labels

    gt_diff = (
        gt_durations[a]
        - gt_durations[b]
    )

    context_diff = (
        context_durations[a]
        - context_durations[b]
    )

    gt_relation = (
        "a_longer"
        if gt_diff > 0.5
        else "b_longer"
        if gt_diff < -0.5
        else "same"
    )

    context_relation = (
        "a_longer"
        if context_diff > 0.5
        else "b_longer"
        if context_diff < -0.5
        else "same"
    )

    if gt_relation != context_relation:

        return (
            f"Comparison perception mismatch: "
            f"ground truth relation={gt_relation}, "
            f"context relation={context_relation}."
        )

    return None


# ============================================================
# ATTRIBUTION
# ============================================================

def attribute(
    result,
    gt_events,
    context_events
):

    question_type = result.get(
        "question_type",
        ""
    )

    question = result.get(
        "question",
        ""
    )

    answer = result.get(
        "answer",
        ""
    )

    prediction = result.get(
        "prediction",
        ""
    )

    context_analysis = analyze_context(
        gt_events,
        context_events
    )

    missing = context_analysis[
        "missing"
    ]

    extra = context_analysis[
        "extra"
    ]

    # --------------------------------------------------------
    # Strong perception checks
    # --------------------------------------------------------

    reason = None

    if question_type == "counting":

        reason = counting_error_reason(
            question,
            answer,
            gt_events,
            context_events
        )

    elif question_type == "duration":

        reason = duration_error_reason(
            question,
            answer,
            gt_events,
            context_events
        )

    elif question_type == "temporal":

        reason = temporal_error_reason(
            question,
            answer,
            gt_events,
            context_events
        )

    elif question_type == "comparison":

        reason = comparison_error_reason(
            question,
            answer,
            gt_events,
            context_events
        )

    # General evidence check.
    if not reason:

        support = context_has_required_labels(
            question,
            answer,
            gt_events,
            context_events
        )

        if support is False:

            reason = (
                "Required audio evidence is "
                "missing from the generated context."
            )

    # --------------------------------------------------------
    # Scope missing/extra events to only those relevant
    # to THIS question.
    # --------------------------------------------------------

    labels = relevant_labels(
        question,
        answer
    )

    relevant_missing = [
        e for e in missing
        if e["label"] in labels
    ]

    relevant_extra = [
        e for e in extra
        if e["label"] in labels
    ]

    # --------------------------------------------------------
    # Attribution
    # --------------------------------------------------------

    if reason:

        attribution = (
            "perception/context"
        )

    elif relevant_missing or relevant_extra:

        attribution = (
            "mixed/ambiguous"
        )

        reason = (
            "The context differs from the "
            "ground truth for a sound relevant "
            "to this question, but the available "
            "evidence does not clearly establish "
            "that this caused the incorrect answer."
        )

    else:

        attribution = (
            "QA reasoning"
        )

        reason = (
            "The required audio evidence is "
            "represented in the context, but "
            "the QA model produced an incorrect answer."
        )

    return {
        "attribution": attribution,
        "reason": reason,
        "missing_events": missing,
        "extra_events": extra,
        "relevant_missing_events": relevant_missing,
        "relevant_extra_events": relevant_extra,
        "matches": context_analysis[
            "matches"
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AUDIO QA ERROR ATTRIBUTION")
    print("=" * 65)

    scored = load_json(
        SCORED_FILE
    )

    print(
        f"Scored predictions: {len(scored)}"
    )

    # Correctness lives inside result["score"]["correct"].
    incorrect = [
        result
        for result in scored
        if result.get(
            "score",
            {}
        ).get(
            "correct"
        ) is False
    ]

    print(
        f"Incorrect predictions: "
        f"{len(incorrect)}"
    )

    counts = {
        "perception/context": 0,
        "QA reasoning": 0,
        "mixed/ambiguous": 0,
    }

    rows = []

    for index, result in enumerate(
        incorrect,
        start=1
    ):

        scene_id = result.get(
            "scene_id"
        )

        question_type = result.get(
            "question_type"
        )

        question = result.get(
            "question"
        )

        answer = result.get(
            "answer",
            ""
        )

        prediction = result.get(
            "prediction",
            ""
        )

        # ----------------------------------------------------
        # Load manifest
        # ----------------------------------------------------

        manifest_path = (
            MANIFEST_DIR /
            f"{scene_id}.json"
        )

        if not manifest_path.exists():

            manifest_path = (
                MANIFEST_DIR /
                f"{scene_id}_manifest.json"
            )

        # ----------------------------------------------------
        # Load context
        # ----------------------------------------------------

        context_path = (
            CONTEXT_DIR /
            f"{scene_id}_context.json"
        )

        if not manifest_path.exists():

            attribution = (
                "perception/context"
            )

            reason = (
                "Ground truth manifest "
                "could not be found."
            )

            counts[
                attribution
            ] += 1

            rows.append({
                "scene_id": scene_id,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "attribution": attribution,
                "reason": reason,
            })

            continue

        if not context_path.exists():

            attribution = (
                "perception/context"
            )

            reason = (
                "Generated context file "
                "could not be found."
            )

            counts[
                attribution
            ] += 1

            rows.append({
                "scene_id": scene_id,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "attribution": attribution,
                "reason": reason,
            })

            continue

        manifest = load_json(
            manifest_path
        )

        context = load_json(
            context_path
        )

        gt_events = (
            extract_manifest_events(
                manifest
            )
        )

        context_events = (
            extract_context_events(
                context
            )
        )

        result_info = attribute(
            result,
            gt_events,
            context_events
        )

        attribution = result_info[
            "attribution"
        ]

        counts[attribution] += 1

        rows.append({
            "scene_id": scene_id,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "attribution": attribution,
            "reason": result_info[
                "reason"
            ],
            "missing_events": result_info[
                "missing_events"
            ],
            "extra_events": result_info[
                "extra_events"
            ],
            "matches": result_info[
                "matches"
            ],
        })

        if index % 20 == 0:

            print(
                f"Processed {index}/"
                f"{len(incorrect)}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(incorrect)

    percentages = {}

    for key, value in counts.items():

        percentages[key] = (
            value / total * 100
            if total
            else 0
        )

    output = {
        "summary": {
            "total_scored": len(scored),
            "incorrect_predictions": total,
            "attribution_counts": counts,
            "attribution_percentages":
                percentages,
        },
        "examples": rows,
    }

    save_json(
        OUTPUT_JSON,
        output
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "question_type",
                "question",
                "answer",
                "prediction",
                "attribution",
                "reason",
            ],
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                key: row.get(
                    key,
                    ""
                )
                for key in writer.fieldnames
            })

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 65)
    print("ERROR ATTRIBUTION RESULTS")
    print("=" * 65)

    print(
        f"Total scored: {len(scored)}"
    )

    print(
        f"Incorrect: {total}"
    )

    print()

    for key, value in counts.items():

        print(
            f"{key}: "
            f"{value} "
            f"({percentages[key]:.1f}%)"
        )

    print()
    print(
        f"Saved JSON: {OUTPUT_JSON}"
    )

    print(
        f"Saved CSV:  {OUTPUT_CSV}"
    )

    # ========================================================
    # EXAMPLES
    # ========================================================

    print()
    print("=" * 65)
    print("EXAMPLE ERRORS")
    print("=" * 65)

    shown = 0

    for row in rows:

        if shown >= 8:
            break

        print()
        print(
            f"[{row['attribution']}] "
            f"{row['scene_id']} | "
            f"{row['question_type']}"
        )

        print(
            f"Q: {row['question']}"
        )

        print(
            f"GT: {row['answer']}"
        )

        print(
            f"Pred: {row['prediction']}"
        )

        print(
            f"Reason: {row['reason']}"
        )

        shown += 1

    # ========================================================
    # QA REASONING EXAMPLES
    # ========================================================

    qa_reasoning_rows = [
        row for row in rows
        if row["attribution"] == "QA reasoning"
    ]

    print()
    print("=" * 65)
    print(
        f"QA REASONING EXAMPLES "
        f"({len(qa_reasoning_rows)} found)"
    )
    print("=" * 65)

    for row in qa_reasoning_rows[:5]:

        print()

        print(
            f"{row['scene_id']} | "
            f"{row['question_type']}"
        )

        print(
            f"Q: {row['question']}"
        )

        print(
            f"GT: {row['answer']}"
        )

        print(
            f"Pred: {row['prediction']}"
        )

        print(
            f"Reason: {row['reason']}"
        )


if __name__ == "__main__":
    main()