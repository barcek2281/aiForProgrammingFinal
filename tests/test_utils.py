"""Tests for the shared helpers in src/utils.py."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.utils import Timer, format_bytes, load_json, save_json


class FakeNumpyScalar:
    """Stands in for a numpy scalar: json cannot serialise it, but it has .item()."""

    def item(self):
        return 7


class TestFormatBytes(unittest.TestCase):
    def test_units(self):
        cases = {0: "0.0 B", 512: "512.0 B", 1024: "1.0 KB",
                 1536: "1.5 KB", 1024 ** 2: "1.0 MB", 1024 ** 3: "1.0 GB"}
        for n_bytes, expected in cases.items():
            with self.subTest(n_bytes=n_bytes):
                self.assertEqual(format_bytes(n_bytes), expected)

    def test_largest_unit_is_gb(self):
        self.assertEqual(format_bytes(1024 ** 4), "1024.0 GB")

    def test_accepts_float(self):
        self.assertEqual(format_bytes(1024.0), "1.0 KB")


class TestJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "report.json"

    def test_roundtrip_creates_missing_folders(self):
        saved = save_json({"rows": 5}, self.path)
        self.assertTrue(saved.exists())
        self.assertEqual(load_json(self.path), {"rows": 5})

    def test_kazakh_text_is_not_escaped(self):
        save_json({"section": "Қаржы"}, self.path)
        self.assertIn("Қаржы", self.path.read_text(encoding="utf-8"))

    def test_numpy_like_and_datetime_are_serialised(self):
        save_json({"n": FakeNumpyScalar(), "when": datetime(2023, 1, 1)}, self.path)
        data = load_json(self.path)
        self.assertEqual(data["n"], 7)
        self.assertEqual(data["when"], "2023-01-01T00:00:00")

    def test_unknown_object_becomes_a_string(self):
        save_json({"path": Path("data") / "raw"}, self.path)
        self.assertIsInstance(load_json(self.path)["path"], str)

    def test_load_json_reports_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_json(self.path)

    def test_written_file_is_valid_json(self):
        save_json([1, 2, 3], self.path)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), [1, 2, 3])


class TestTimer(unittest.TestCase):
    def test_measures_a_non_negative_time(self):
        with Timer() as timer:
            sum(range(1000))
        self.assertGreaterEqual(timer.seconds, 0.0)

    def test_exceptions_are_not_swallowed(self):
        with self.assertRaises(ValueError):
            with Timer():
                raise ValueError("boom")


if __name__ == "__main__":
    unittest.main()
