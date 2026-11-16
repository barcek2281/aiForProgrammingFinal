"""Tests for src/config.py: path helpers and the column groups the audit relies on."""
import tempfile
import unittest
from pathlib import Path

from src import config


class TestProjectPath(unittest.TestCase):
    def test_a_path_inside_the_project_becomes_relative(self):
        self.assertEqual(config.project_path(config.RAW_CSV),
                         "data/raw/data/kazakhlawcorpus_kk_only.csv")

    def test_the_separator_is_always_a_forward_slash(self):
        # the same report must be produced on Windows and on macOS
        self.assertNotIn("\\", config.project_path(config.PROCESSED_DIR / "acts_full.parquet"))

    def test_the_project_root_itself(self):
        self.assertEqual(config.project_path(config.BASE_DIR), ".")

    def test_accepts_a_string(self):
        self.assertEqual(config.project_path(str(config.SAMPLE_CSV)),
                         "data/sample/acts_sample_500.csv")

    def test_a_path_outside_the_project_is_returned_unchanged(self):
        outside = Path(tempfile.gettempdir()) / "acts.csv"
        self.assertEqual(config.project_path(outside), str(outside))


class TestGetPaths(unittest.TestCase):
    def test_full_data(self):
        paths = config.get_paths(use_sample=False)
        self.assertEqual(paths["name"], "full")
        self.assertEqual(paths["csv"], config.RAW_CSV)
        self.assertEqual(paths["parquet"].name, "acts_full.parquet")
        self.assertEqual(paths["reports"], config.REPORTS_DIR / "full")

    def test_sample_data(self):
        paths = config.get_paths(use_sample=True)
        self.assertEqual(paths["name"], "sample")
        self.assertEqual(paths["csv"], config.SAMPLE_CSV)
        self.assertEqual(paths["parquet"].name, "acts_sample.parquet")

    def test_the_two_modes_never_share_a_file(self):
        full, sample = config.get_paths(False), config.get_paths(True)
        self.assertNotEqual(full["parquet"], sample["parquet"])
        self.assertNotEqual(full["reports"], sample["reports"])


class TestColumnGroups(unittest.TestCase):
    def test_multivalue_columns_are_also_categories(self):
        for column in config.MULTIVALUE_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, config.CATEGORY_COLUMNS)

    def test_placeholders_are_lower_case(self):
        # they are compared against value.strip().lower()
        for placeholder in config.PLACEHOLDERS:
            with self.subTest(placeholder=placeholder):
                self.assertEqual(placeholder, placeholder.lower())

    def test_the_target_and_the_text_column_are_not_mixed_up(self):
        self.assertEqual(config.TEXT_COLUMN, "title")
        self.assertIn("database_section", config.CATEGORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
