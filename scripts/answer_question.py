import json
import os
import sys

from groq import Groq


MODEL_NAME = "openai/gpt-oss-20b"


def load_context(path):
    """Load a context JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(context):
    """Convert structured audio context into readable text."""

    lines = []

    audio = context.get("audio", {})

    lines.append(
        f"Audio duration: {audio.get('duration', 'unknown')} seconds"
    )

    lines.append("")
    lines.append("Detected audio events:")

    events = context.get("events", [])

    if not events:
        lines.append("No detected events.")
        return "\n".join(lines)

    for event in events:
        lines.append(
            f"- {event['event_id']}: "
            f"{event['label']} "
            f"from {event['start']:.2f}s to {event['end']:.2f}s "
            f"(duration {event['duration']:.2f}s, "
            f"confidence {event['confidence']:.3f})"
        )

    return "\n".join(lines)


def build_prompt(context, question):
    """Build the QA prompt using only the detected audio context."""

    context_text = build_context(context)

    prompt = f"""
You are answering a question about an audio recording.

You MUST answer using only the detected audio context provided below.

Do not assume that an event occurred if it is not present in the context.

For counting questions:
- Count separate event IDs.
- Do not count overlapping evidence as separate events unless they correspond to separate event IDs.

For temporal questions:
- Use the timestamps.
- Compare event start and end times carefully.

For duration questions:
- Use the provided event durations.

For comparison questions:
- Compare only events supported by the context.

For apparent or perception questions:
- Report only sounds represented in the detected audio events.

For yes or no questions:
- Answer only from the available context.
- If the context does not provide enough evidence, say that it cannot be determined.

For causal or reasoning questions:
- Audio evidence does not prove real world causality.
- Give an evidence based interpretation.
- Use wording such as "may be because", "could be related to", or "may indicate" when appropriate.
- Do not present an inferred cause as a confirmed fact.

If the context does not contain enough information to answer the question, use:
"Cannot determine from the available audio context."

Return ONLY a JSON object using this exact structure:

{{
  "answer": "your answer",
  "confidence": 0.0,
  "evidence_event_ids": []
}}

The confidence must be a number between 0 and 1.

The evidence_event_ids list must contain the IDs of the events used to answer the question.

Question:
{question}

Audio context:
{context_text}
"""

    return prompt


def ask_groq(prompt):
    """Send the prompt to Groq."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Set the API key in PowerShell before running this script."
        )

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return completion.choices[0].message.content


def clean_json_response(response):
    """Clean markdown code fences from the model response."""

    cleaned = response.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def parse_response(response):
    """Parse and validate the model response."""

    cleaned = clean_json_response(response)

    try:
        parsed = json.loads(cleaned)

        required_fields = [
            "answer",
            "confidence",
            "evidence_event_ids"
        ]

        for field in required_fields:
            if field not in parsed:
                raise ValueError(
                    f"Groq response is missing required field: {field}"
                )

        if not isinstance(parsed["confidence"], (int, float)):
            raise ValueError("confidence must be a number")

        if not 0 <= parsed["confidence"] <= 1:
            raise ValueError("confidence must be between 0 and 1")

        if not isinstance(parsed["evidence_event_ids"], list):
            raise ValueError("evidence_event_ids must be a list")

        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        print("\nWarning: Groq returned invalid JSON.")
        print("\nRaw response:")
        print(response)
        print(f"\nParsing error: {e}")

        return None


def main():

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            'python scripts/answer_question.py '
            '<context_json> "<question>"'
        )
        sys.exit(1)

    context_path = sys.argv[1]
    question = sys.argv[2]

    if not os.path.exists(context_path):
        print(f"Error: Context file not found: {context_path}")
        sys.exit(1)

    context = load_context(context_path)

    prompt = build_prompt(
        context=context,
        question=question
    )

    print("\nQuestion:")
    print(question)

    print("\nSending context to Groq...\n")

    try:
        raw_response = ask_groq(prompt)

    except Exception as e:
        print("\nError while calling Groq:")
        print(e)
        sys.exit(1)

    print("Groq response:")
    print(raw_response)

    parsed = parse_response(raw_response)

    if parsed is None:
        sys.exit(1)

    print("\nParsed answer:")
    print(json.dumps(parsed, indent=2))

    print("\nAnswer:")
    print(parsed["answer"])

    print("\nConfidence:")
    print(parsed["confidence"])

    print("\nEvidence:")
    if parsed["evidence_event_ids"]:
        for event_id in parsed["evidence_event_ids"]:
            print(f"- {event_id}")
    else:
        print("- None")


if __name__ == "__main__":
    main()
