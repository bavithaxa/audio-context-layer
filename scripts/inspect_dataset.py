import csv
from pathlib import Path
from collections import Counter

# Location of the ESC 50 metadata file
csv_path = Path("ESC-50-master/meta/esc50.csv")

# Read the dataset
with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Total audio clips:", len(rows))

# Count clips in each category
category_counts = Counter(row["category"] for row in rows)

print("\nNumber of categories:", len(category_counts))

print("\nCategories:")
for category, count in sorted(category_counts.items()):
    print(f"{category}: {count}")