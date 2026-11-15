"""Build a small sample (500 acts) without downloading the 337 MB file.

Rows are requested from the public Hugging Face dataset-viewer API at several
offsets and saved as data/sample/acts_sample_500.csv. The sample lets anyone
run the whole pipeline in seconds:  python main.py --sample

Usage:  python scripts/fetch_sample.py
"""
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

API_URL = "https://datasets-server.huggingface.co/rows"
OFFSETS = [0, 40_000, 90_000, 140_000, 190_000]
ROWS_PER_REQUEST = 100


def fetch_rows(offset: int, length: int) -> list:
    """Request `length` rows starting at `offset`; returns a list of dicts."""
    query = urllib.parse.urlencode({
        "dataset": config.HF_REPO_ID, "config": "default", "split": "train",
        "offset": offset, "length": length,
    })
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=60) as response:
        payload = json.load(response)
    return [item["row"] for item in payload["rows"]]


def main() -> None:
    rows = []
    for offset in OFFSETS:
        try:
            part = fetch_rows(offset, ROWS_PER_REQUEST)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"Could not fetch rows at offset {offset}: {error}")
            continue
        rows.extend(part)
        print(f"offset {offset}: {len(part)} rows")

    if not rows:
        print("No rows received - check the internet connection.")
        return

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            # the API returns JSON null for missing values -> empty CSV cell
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    print(f"Saved {len(rows)} rows to {config.SAMPLE_CSV}")


if __name__ == "__main__":
    main()
