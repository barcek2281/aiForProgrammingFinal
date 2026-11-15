"""Small helpers shared by all modules: timing, formatting, JSON files."""
import json
import time
from datetime import datetime
from pathlib import Path


class Timer:
    """Context manager that measures wall-clock time in seconds.

    Example:
        with Timer() as t:
            do_work()
        print(t.seconds)
    """

    def __enter__(self):
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc_info):
        self.seconds = time.perf_counter() - self.start
        return False


def format_bytes(n_bytes: float) -> str:
    """Turn a number of bytes into a readable string, e.g. '336.9 MB'."""
    size = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _json_default(obj):
    """Convert numpy / pandas / datetime objects that json cannot serialise."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "isoformat"):     # datetime, Timestamp
        return obj.isoformat()
    return str(obj)


def save_json(data, path: Path) -> Path:
    """Write a dict / list to a UTF-8 JSON file, creating folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def load_json(path: Path):
    """Read a JSON file. Raises FileNotFoundError if it does not exist."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str) -> None:
    """Print a visible header in the console."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def now_str() -> str:
    """Current time as a string for report metadata."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
