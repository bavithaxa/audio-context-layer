import json
import os
import time
from pathlib import Path

from groq import Groq


# ============================================================
# CONFIG
# ============================================================

QA_FILE = Path("data/qa/qa_all.json")
CONTEXT_DIR = Path("data/context")
RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "qa_predictions.json"

MODEL_NAME = "openai/gpt-oss-20b"

# IMPORTANT:
# 25 test scenes / 5 scenes per request = 5 API calls
SCENES_PER_REQUEST = 5

REQUEST_DELAY = 2


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set.\n\n"
        "Run this in PowerShell:\n"
        '$env:GROQ_API_KEY="YOUR_NEW_KEY"'
    )

client = Groq(api_key=api_key)


# Safety check
if "gemini" in MODEL_NAME.lower():
    raise RuntimeError(
        "STOP: Gemini model detected. This script must use Groq."
    )


# ============================================================
# JSON HELPERS
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
# LOAD QA
# ============================================================

def load_questions():

    data = load_json(QA_FILE)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in ["questions", "qa", "data"]:

            if key in data:
                return data[key]

    raise ValueError(
        "Could not find QA list in qa_all.json"
    )


# ============================================================
# LOAD CONTEXT
# ============================================================

def load_context(scene_id):

    path = (
        CONTEXT_DIR /
        f"{scene_id}_context.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Context file not found: {path}"
        )

    return load_json(path)


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(context):

    audio = context.get(
        "audio",
        {}
    )

    events = context.get(
        "events",
        []
    )

    duration = audio.get(
        "duration",
        "unknown"
    )

    lines = [
        f"Audio duration: {duration} seconds",
        "",
        "Detected events:"
    ]

    if not events:

        lines.append(
            "No events were detected."
        )

    else:

        for event in events:

            event_id = event.get(
                "event_id",
                ""
            )

            label = event.get(
                "label",
                ""
            )

            start = event.get(
                "start",
                0
            )

            end = event.get(
                "end",
                0
            )

            event_duration = event.get(
                "duration",
                0
            )

            confidence = event.get(
                "confidence",
                0
            )

            lines.append(
                f"{event_id}: "
                f"{label}, "
                f"{start:.2f}s to {end:.2f}s, "
                f"duration {event_duration:.2f}s, "
                f"confidence {confidence:.3f}"
            )

    return "\n".join(lines)


# ============================================================
# BUILD ONE LARGE BATCH PROMPT
# ============================================================

def build_batch_prompt(batch_scenes, scene_groups):

    scene_sections = []

    for scene_id in batch_scenes:

        context = load_context(
            scene_id
        )

        context_text = format_context(
            context
        )

        questions = scene_groups[
            scene_id
        ]

        question_lines = []

        for question in questions:

            question_lines.append(
                f"""
QUESTION ID: {question["_eval_id"]}
TYPE: {question.get("question_type", "")}
QUESTION: {question.get("question", "")}
""".strip()
            )

        questions_text = "\n\n".join(
            question_lines
        )

        scene_sections.append(
            f"""
============================================================
SCENE: {scene_id}
============================================================

AUDIO CONTEXT
-------------
{context_text}

QUESTIONS
---------
{questions_text}
""".strip()
        )

    all_scenes_text = "\n\n".join(
        scene_sections
    )

    prompt = f"""
You are evaluating an Audio Question Answering system.

You are given structured audio context for multiple audio scenes.

Your job is to answer every question using ONLY the
corresponding scene's audio context.

IMPORTANT RULES:

1. Never use information from another scene.
2. Never invent sounds that are not present in the context.
3. For counting questions, count detected events.
4. For temporal questions, compare event timestamps.
5. For duration questions, compare event durations.
6. For yes/no questions, answer from the detected events.
7. For comparison questions, compare the relevant event counts.
8. For apparent questions, list the sounds detected.
9. For causal reasoning questions, give only a cautious possible
   explanation based on the available audio evidence.
10. Audio does not prove real world causality. Use words such as
    "may", "might", or "could".
11. If the context does not provide enough information, say:
    "Cannot determine from the available audio context."

Return ONLY valid JSON.

Required JSON structure:

{{
  "answers": [
    {{
      "id": "question_id",
      "answer": "concise answer",
      "confidence": 0.0,
      "evidence_event_ids": ["event_001"]
    }}
  ]
}}

Rules:

- Include exactly one answer object for every question.
- "id" must exactly match the question ID.
- "answer" must be concise and natural.
- "confidence" must be a number between 0 and 1.
- "evidence_event_ids" must contain only event IDs that support
  the answer.
- If there is no supporting event, use [].
- Do not include explanations outside the JSON.

Here are the scenes and questions:

{all_scenes_text}
"""

    return prompt


# ============================================================
# GROQ API CALL
# ============================================================

def call_groq(prompt):

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Audio Question "
                    "Answering evaluator. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_completion_tokens=12000,

        response_format={
            "type": "json_object"
        }
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise ValueError(
            "Groq returned an empty response."
        )

    return content


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_response(text):

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    data = json.loads(text)

    if "answers" not in data:

        raise ValueError(
            "Groq response does not contain 'answers'."
        )

    if not isinstance(
        data["answers"],
        list
    ):

        raise ValueError(
            "'answers' is not a list."
        )

    return data["answers"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AUDIO QA EVALUATION")
    print("=" * 65)

    print(
        f"Provider: Groq"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"QA file: {QA_FILE}"
    )

    print(
        f"Context directory: {CONTEXT_DIR}"
    )

    print()

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    all_questions = load_questions()

    test_questions = [
        q
        for q in all_questions
        if q.get("split") == "test"
    ]

    print(
        f"Total QA pairs: {len(all_questions)}"
    )

    print(
        f"Test QA pairs: {len(test_questions)}"
    )

    if not test_questions:

        raise RuntimeError(
            "No test questions found."
        )

    # --------------------------------------------------------
    # GIVE EVERY QUESTION AN ID
    # --------------------------------------------------------

    for index, question in enumerate(
        test_questions,
        start=1
    ):

        question["_eval_id"] = (
            f"q_{index:03d}"
        )

    # --------------------------------------------------------
    # GROUP QUESTIONS BY SCENE
    # --------------------------------------------------------

    scene_groups = {}

    for question in test_questions:

        scene_id = question[
            "scene_id"
        ]

        if scene_id not in scene_groups:

            scene_groups[
                scene_id
            ] = []

        scene_groups[
            scene_id
        ].append(question)

    scene_ids = sorted(
        scene_groups.keys()
    )

    print(
        f"Test scenes: {len(scene_ids)}"
    )

    print()

    # --------------------------------------------------------
    # CHECK ALL CONTEXT FILES
    # --------------------------------------------------------

    print(
        "Checking context files..."
    )

    missing = []

    for scene_id in scene_ids:

        path = (
            CONTEXT_DIR /
            f"{scene_id}_context.json"
        )

        if not path.exists():

            missing.append(
                str(path)
            )

    if missing:

        print()

        print(
            "ERROR: Missing context files:"
        )

        for path in missing:
            print(path)

        raise RuntimeError(
            f"{len(missing)} context files missing."
        )

    print(
        "All test context files found."
    )

    print()

    # --------------------------------------------------------
    # CREATE SCENE BATCHES
    # --------------------------------------------------------

    batches = []

    for start in range(
        0,
        len(scene_ids),
        SCENES_PER_REQUEST
    ):

        batch = scene_ids[
            start:
            start + SCENES_PER_REQUEST
        ]

        batches.append(
            batch
        )

    print(
        f"Scenes per Groq request: "
        f"{SCENES_PER_REQUEST}"
    )

    print(
        f"Total Groq requests: "
        f"{len(batches)}"
    )

    print()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # PROCESS EACH BATCH
    # --------------------------------------------------------

    for batch_number, batch_scenes in enumerate(
        batches,
        start=1
    ):

        print("=" * 65)

        print(
            f"GROQ REQUEST "
            f"{batch_number}/{len(batches)}"
        )

        print(
            "Scenes: "
            + ", ".join(batch_scenes)
        )

        print("=" * 65)

        try:

            # --------------------------------------------
            # Build ONE prompt containing all 5 scenes
            # --------------------------------------------

            prompt = build_batch_prompt(
                batch_scenes,
                scene_groups
            )

            total_questions = sum(
                len(
                    scene_groups[
                        scene_id
                    ]
                )
                for scene_id in batch_scenes
            )

            print(
                f"Questions in this request: "
                f"{total_questions}"
            )

            print(
                "Sending ONE request to Groq..."
            )

            raw_response = call_groq(
                prompt
            )

            answers = parse_response(
                raw_response
            )

            print(
                f"Groq returned "
                f"{len(answers)} answers."
            )

            # --------------------------------------------
            # Map answers by ID
            # --------------------------------------------

            answer_map = {}

            for answer in answers:

                answer_id = str(
                    answer.get("id", "")
                )

                if answer_id:

                    answer_map[
                        answer_id
                    ] = answer

            # --------------------------------------------
            # Store results
            # --------------------------------------------

            for scene_id in batch_scenes:

                for question in scene_groups[
                    scene_id
                ]:

                    qid = question[
                        "_eval_id"
                    ]

                    model_answer = (
                        answer_map.get(qid)
                    )

                    result = {
                        "eval_id": qid,
                        "scene_id": scene_id,
                        "question_type": question.get(
                            "question_type",
                            ""
                        ),
                        "question": question.get(
                            "question",
                            ""
                        ),
                        "ground_truth": question.get(
                            "answer",
                            question.get(
                                "ground_truth",
                                ""
                            )
                        ),
                        "prediction": None,
                        "confidence": None,
                        "evidence_event_ids": [],
                        "error": None
                    }

                    if model_answer is None:

                        result["error"] = (
                            "Question ID missing "
                            "from Groq response."
                        )

                    else:

                        result["prediction"] = (
                            model_answer.get(
                                "answer"
                            )
                        )

                        result["confidence"] = (
                            model_answer.get(
                                "confidence"
                            )
                        )

                        result[
                            "evidence_event_ids"
                        ] = model_answer.get(
                            "evidence_event_ids",
                            []
                        )

                    results.append(
                        result
                    )

            print(
                f"Batch {batch_number} completed."
            )

        except Exception as e:

            print()
            print(
                f"ERROR in batch "
                f"{batch_number}:"
            )
            print(e)
            print()

            # --------------------------------------------
            # Preserve errors for this batch
            # --------------------------------------------

            for scene_id in batch_scenes:

                for question in scene_groups[
                    scene_id
                ]:

                    results.append(
                        {
                            "eval_id": question[
                                "_eval_id"
                            ],
                            "scene_id": scene_id,
                            "question_type": question.get(
                                "question_type",
                                ""
                            ),
                            "question": question.get(
                                "question",
                                ""
                            ),
                            "ground_truth": question.get(
                                "answer",
                                question.get(
                                    "ground_truth",
                                    ""
                                )
                            ),
                            "prediction": None,
                            "confidence": None,
                            "evidence_event_ids": [],
                            "error": str(e)
                        }
                    )

            # Save immediately
            save_json(
                OUTPUT_FILE,
                results
            )

            print(
                f"Partial results saved to:"
            )

            print(
                OUTPUT_FILE
            )

            # IMPORTANT:
            # Do not automatically retry 429 errors.
            print()
            print(
                "Stopping here so we do not "
                "waste additional API requests."
            )

            break

        # --------------------------------------------
        # Save after EVERY successful batch
        # --------------------------------------------

        save_json(
            OUTPUT_FILE,
            results
        )

        print(
            f"Progress saved: "
            f"{len(results)} questions"
        )

        # Wait before next API request
        if batch_number < len(batches):

            print(
                f"Waiting {REQUEST_DELAY} seconds..."
            )

            time.sleep(
                REQUEST_DELAY
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    successful = sum(
        1
        for result in results
        if (
            result["prediction"]
            is not None
            and result["error"]
            is None
        )
    )

    errors = (
        len(results)
        - successful
    )

    print()
    print("=" * 65)
    print("EVALUATION FINISHED")
    print("=" * 65)

    print(
        f"Results generated: "
        f"{len(results)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Errors: "
        f"{errors}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print()

    # --------------------------------------------------------
    # QUESTION TYPE SUMMARY
    # --------------------------------------------------------

    type_counts = {}

    for result in results:

        qtype = result[
            "question_type"
        ]

        if qtype not in type_counts:

            type_counts[qtype] = {
                "total": 0,
                "successful": 0
            }

        type_counts[
            qtype
        ]["total"] += 1

        if (
            result["prediction"]
            is not None
            and result["error"]
            is None
        ):

            type_counts[
                qtype
            ]["successful"] += 1

    print(
        "Question type breakdown:"
    )

    for qtype, counts in sorted(
        type_counts.items()
    ):

        print(
            f"  {qtype}: "
            f"{counts['successful']}/"
            f"{counts['total']}"
        )


if __name__ == "__main__":
    main()