"""Labelled data for the models and a train/test split without leakage.

Two kinds of leakage are prevented:
  1. time    - the test set contains only acts adopted after the training period,
               exactly as the model would be used on new acts;
  2. copies  - many acts share the same title; a test title that already occurs
               in the training period is removed, otherwise the model would be
               rewarded for remembering it.
Only the title is used as a feature. 'description' is not loaded at all: it
contains the act form and number in plain text.
"""
import pandas as pd

from src.audit import normalize_title
from src.db import run_query
from src.queries import SPLIT_DATE

LABELLED_SQL = """
    SELECT a.act_id, a.title, a.date_adopted, s.name AS section
    FROM acts a
    JOIN database_sections s ON s.section_id = a.section_id
    ORDER BY a.date_adopted, a.act_id;
"""


def load_labelled(db_path) -> pd.DataFrame:
    """Acts that have a section (the target), read from SQLite."""
    df = run_query(db_path, LABELLED_SQL)
    df["title_key"] = normalize_title(df["title"])
    return df


def time_split(df: pd.DataFrame, split_date: str = SPLIT_DATE) -> tuple:
    """Return (train, test, report). Test = adopted from split_date, unseen titles only."""
    train = df[df["date_adopted"] < split_date]
    test_all = df[df["date_adopted"] >= split_date]
    seen_in_train = test_all["title_key"].isin(set(train["title_key"]))
    test = test_all[~seen_in_train]

    train_classes, test_classes = set(train["section"]), set(test["section"])
    report = {
        "split_date": split_date,
        "train_rows": len(train),
        "test_rows_before_cleaning": len(test_all),
        "test_rows_removed_title_seen_in_train": int(seen_in_train.sum()),
        "test_rows": len(test),
        "test_share_pct": round(100 * len(test) / max(len(train) + len(test), 1), 1),
        "train_date_range": [train["date_adopted"].min(), train["date_adopted"].max()],
        "test_date_range": [test["date_adopted"].min(), test["date_adopted"].max()],
        "classes_train": len(train_classes),
        "classes_test": len(test_classes),
        "classes_only_in_test": sorted(test_classes - train_classes),
        "titles_shared_between_train_and_test": len(set(train["title_key"]) & set(test["title_key"])),
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), report


def validation_split(train: pd.DataFrame, months: int = 12) -> tuple:
    """Split the training period again by time: the last `months` are for validation.

    Hyper-parameters are chosen on this validation part, never on the test set.
    """
    last_date = pd.Timestamp(train["date_adopted"].max())
    border = (last_date - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
    fit_part = train[train["date_adopted"] <= border]
    validation_all = train[train["date_adopted"] > border]
    validation = validation_all[~validation_all["title_key"].isin(set(fit_part["title_key"]))]
    return fit_part, validation, border
