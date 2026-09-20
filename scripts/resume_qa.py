import os
import json
import time
from pathlib import Path

from groq import Groq


# ============================================================
# CONFIG
# ============================================================

QA_FILE = Path("data/qa/qa_all.json")
CONTEXT_DIR = Path("data/context")
RESULTS_FILE = Path("results/qa_predictions.json")

MODEL_NAME = "openai/gpt-oss-20b"

# One question per request. This removes any possibility of
# answers being mismatched to the wrong question due to
# ordering or missing IDs in a batched response.
QUESTIONS_PER_REQUEST = 1

REQUEST_DELAY_SECONDS = 1.5


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

client = Groq(api_key=api_key)


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


def load_json_or_empty(path):
    if not path.exists():
        return []
    return load_json(path)


def get_questions(data):

    if isinstance(data, list):
        return data

    for key in ["questions", "qa", "data"]:
        if key in data:
            return data[key]

    raise ValueError(
        "Could not find QA list in QA file."
    )


# ============================================================
# RESULT IDENTIFICATION
# ============================================================

def make_key(item):

    return (
        item.get("scene_id"),
        item.get("question_type"),
        item.get("question")
    )


def is_successful(result):

    if result.get("error"):
        return False

    prediction = result.get("prediction")

    return prediction is not None


# ============================================================
# CONTEXT
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


def format_context(context):

    audio = context.get("audio", {})
    events = context.get("events", [])

    lines = []

    duration = audio.get(
        "duration",
        "unknown"
    )

    lines.append(
        f"Audio duration: {duration} seconds"
    )

    lines.append(
        "Detected events:"
    )

    if not events:
        lines.append("No detected events.")

    for event in events:

        start = float(
            event.get("start", 0)
        )

        end = float(
            event.get("end", 0)
        )

        event_duration = float(
            event.get("duration", end - start)
        )

        confidence = float(
            event.get("confidence", 0)
        )

        lines.append(
            f"- {event.get('event_id')}: "
            f"{event.get('label')} "
            f"{start:.2f}-{end:.2f}s "
            f"(duration {event_duration:.2f}s, "
            f"confidence {confidence:.3f})"
        )

    return "\n".join(lines)


# ============================================================
# PROMPT (single question)
# ============================================================

def build_prompt(question):

    scene_id = question["scene_id"]

    context = load_context(scene_id)

    context_text = format_context(context)

    return f"""
You are answering a question about an audio recording.

Use ONLY the supplied audio context below. Do not use outside
knowledge and do not invent events that are not listed.

Rules by question type:

1. Counting questions:
   Count the distinct detected events shown in the context.

2. Temporal questions:
   Compare event start and end timestamps carefully.

3. Duration questions:
   Use the duration values shown in the context.

4. Comparison questions:
   Compare only the specific events named in the question,
   using their timestamps/durations from the context.

5. Apparent questions:
   List only sounds that are represented in the detected events.

6. Yes/no questions:
   Answer strictly from the context. If the evidence is not
   present, say it cannot be determined.

7. Causal/reasoning questions:
   Audio evidence does not prove real-world causality. Use
   cautious wording such as "may be because" or "could indicate".
   Do not state an inferred cause as a confirmed fact.

8. If the context does not contain enough information, answer:
   "Cannot determine from the available audio context."

9. Include the event IDs you used as evidence.

10. Confidence must be a number between 0 and 1.

Return ONLY a single JSON object, with no markdown fences and
no extra text, in exactly this structure:

{{
  "answer": "your answer",
  "confidence": 0.0,
  "evidence_event_ids": []
}}

QUESTION TYPE:
{question.get("question_type")}

AUDIO CONTEXT:
{context_text}

QUESTION:
{question.get("question")}
"""


# ============================================================
# STRICT RESPONSE PARSER
# ============================================================

REQUIRED_FIELDS = [
    "answer",
    "confidence",
    "evidence_event_ids"
]


def parse_groq_response(raw_text):

    if not raw_text:
        raise ValueError("Groq returned an empty response.")

    text = raw_text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]
    elif "```JSON" in text:
        text = text.split("```JSON", 1)[1]

    if "```" in text:
        text = text.split("```", 1)[0]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Groq response did not contain a JSON object.\n\n"
            f"Raw response:\n{raw_text}"
        )

    text = text[start:end + 1]

    parsed = json.loads(text)

    # Strict validation: every required field must be present.
    # We do NOT guess or fall back if something is missing --
    # a missing field means this question is marked as failed
    # and can be retried, rather than silently mismatched.
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]

    if missing:
        raise ValueError(
            f"Response missing required fields: {missing}\n\n"
            f"Raw response:\n{raw_text}"
        )

    return parsed


# ============================================================
# LOAD DATA
# ============================================================

qa_data = load_json(QA_FILE)
all_qa = get_questions(qa_data)

test_qa = [
    q for q in all_qa
    if q.get("split") == "test"
]

existing_results = load_json_or_empty(RESULTS_FILE)

print("=" * 65)
print("AUDIO QA EVALUATION (single-question mode)")
print("=" * 65)

print(f"Total test QA: {len(test_qa)}")
print(f"Existing results found: {len(existing_results)}")


# ============================================================
# INDEX EXISTING RESULTS
# ============================================================

result_map = {}

for result in existing_results:
    result_map[make_key(result)] = result


# ============================================================
# FIND REMAINING QUESTIONS
# ============================================================

remaining = []

for question in test_qa:

    key = make_key(question)
    existing = result_map.get(key)

    if existing is None:
        remaining.append(question)
    elif not is_successful(existing):
        remaining.append(question)


successful_count = sum(
    1 for r in result_map.values() if is_successful(r)
)

print(f"Successful already: {successful_count}")
print(f"Questions remaining: {len(remaining)}")

if not remaining:
    print("\nAll questions already have successful results.")
    raise SystemExit


# ============================================================
# MAIN LOOP - ONE QUESTION AT A TIME
# ============================================================

total = len(remaining)

for index, question in enumerate(remaining, start=1):

    key = make_key(question)

    print()
    print("-" * 65)
    print(f"[{index}/{total}] {question.get('scene_id')} | "
          f"{question.get('question_type')}")
    print("-" * 65)

    try:
        prompt = build_prompt(question)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_text = response.choices[0].message.content

        parsed = parse_groq_response(raw_text)

        result_map[key] = {
            **question,
            "prediction": parsed.get("answer"),
            "confidence": parsed.get("confidence"),
            "evidence_event_ids": parsed.get(
                "evidence_event_ids", []
            ),
            "error": None
        }

        print(f"OK: {parsed.get('answer')}")

    except Exception as e:

        print(f"FAILED: {e}")

        result_map[key] = {
            **question,
            "prediction": None,
            "confidence": None,
            "evidence_event_ids": [],
            "error": str(e)
        }

    # Save after every single question so progress is never lost.
    save_json(RESULTS_FILE, list(result_map.values()))

    if index < total:
        time.sleep(REQUEST_DELAY_SECONDS)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_results = list(result_map.values())

successful = sum(1 for r in final_results if is_successful(r))
errors = len(final_results) - successful

save_json(RESULTS_FILE, final_results)

print()
print("=" * 65)
print("EVALUATION FINISHED")
print("=" * 65)
print(f"Results: {len(final_results)}")
print(f"Successful: {successful}")
print(f"Errors: {errors}")
print(f"Saved to: {RESULTS_FILE}")