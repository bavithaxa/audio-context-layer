import json
from pathlib import Path

path = Path("results/qa_predictions.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

types = [
    "counting",
    "temporal",
    "comparison",
    "duration",
    "yes_no",
    "apparent",
    "causal_reasoning",
]

for qtype in types:

    print("\n" + "=" * 70)
    print(qtype.upper())
    print("=" * 70)

    items = [
        x for x in data
        if x.get("question_type") == qtype
    ]

    for item in items[:5]:

        print()
        print("QUESTION:")
        print(item.get("question"))

        print("GROUND TRUTH:")
        print(item.get("ground_truth"))

        print("PREDICTION:")
        print(item.get("prediction"))

        print("-" * 50)