"""Relational schema in SQLite: normalisation of the flat table and loading.

The flat Parquet table repeats the same strings in every row and keeps several
values in one cell. Here it is split into:

    acts                 one row per legal act
    database_sections    lookup: thematic section (the classification target)
    legal_forces         lookup: legal force of the act
    regions              lookup: region
    areas  + act_areas   areas of legal regulation   (many-to-many)
    forms  + act_forms   forms of the act            (many-to-many)
    bodies + act_bodies  adopting bodies             (many-to-many)
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src import config
from src.audit import split_multivalue
from src.data_loader import load_acts
from src.utils import Timer

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE database_sections (
    section_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE legal_forces (
    legal_force_id  INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE regions (
    region_id       INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE areas (
    area_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE forms (
    form_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE bodies (
    body_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    history_note    TEXT                    -- text in parentheses: renamed / abolished ...
);

CREATE TABLE acts (
    act_id          INTEGER PRIMARY KEY,    -- source_id of the corpus
    law_id          TEXT NOT NULL UNIQUE,   -- code used by adilet.zan.kz
    title           TEXT NOT NULL,
    date_adopted    TEXT NOT NULL,          -- ISO date YYYY-MM-DD
    date_modified   TEXT,
    date_published  TEXT,
    is_repealed     INTEGER NOT NULL CHECK (is_repealed IN (0, 1)),
    section_id      INTEGER REFERENCES database_sections(section_id),
    legal_force_id  INTEGER REFERENCES legal_forces(legal_force_id),
    region_id       INTEGER REFERENCES regions(region_id)
);

CREATE TABLE act_areas (
    act_id          INTEGER NOT NULL REFERENCES acts(act_id),
    area_id         INTEGER NOT NULL REFERENCES areas(area_id),
    PRIMARY KEY (act_id, area_id)
);
CREATE TABLE act_forms (
    act_id          INTEGER NOT NULL REFERENCES acts(act_id),
    form_id         INTEGER NOT NULL REFERENCES forms(form_id),
    PRIMARY KEY (act_id, form_id)
);
CREATE TABLE act_bodies (
    act_id          INTEGER NOT NULL REFERENCES acts(act_id),
    body_id         INTEGER NOT NULL REFERENCES bodies(body_id),
    PRIMARY KEY (act_id, body_id)
);

CREATE INDEX idx_acts_date     ON acts(date_adopted);
CREATE INDEX idx_acts_section  ON acts(section_id);
CREATE INDEX idx_acts_title    ON acts(title);
CREATE INDEX idx_act_areas_area   ON act_areas(area_id);
CREATE INDEX idx_act_forms_form   ON act_forms(form_id);
CREATE INDEX idx_act_bodies_body  ON act_bodies(body_id);
"""

# flat columns needed to build the database
SOURCE_COLUMNS = [
    "source_id", "law_id", "title", "date_adopted", "date_modified", "date_published",
    "is_repealed", "database_section", "legal_force", "region",
    "legal_relation_area", "act_form", "adopting_body",
]
TABLE_ORDER = ["database_sections", "legal_forces", "regions", "areas", "forms", "bodies",
               "acts", "act_areas", "act_forms", "act_bodies"]


# --------------------------------------------------------------------------
# Building the tables with pandas
# --------------------------------------------------------------------------
def build_lookup(values: pd.Series, id_column: str) -> pd.DataFrame:
    """Lookup table (id, name) from the distinct non-missing values of a column."""
    names = sorted(values.dropna().astype(str).unique())
    return pd.DataFrame({id_column: range(1, len(names) + 1), "name": names})


def to_ids(values: pd.Series, lookup: pd.DataFrame, id_column: str) -> pd.Series:
    """Replace names by their ids from a lookup table (missing stays missing)."""
    mapping = dict(zip(lookup["name"], lookup[id_column]))
    return values.astype("object").map(mapping).astype("Int64")


def split_body(value: str) -> tuple:
    """'Ministry X (renamed in 2014; ...)' -> ('Ministry X', 'renamed in 2014; ...')."""
    match = re.match(r"^(.*?)\s*\((.*)\)\s*$", value, flags=re.DOTALL)
    if match and match.group(1):
        return match.group(1).strip(), match.group(2).strip()
    return value.strip(), None


def explode_multivalue(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per (act_id, value) for a multi-valued column."""
    cells = df[column].astype("object")
    split_cache = {cell: split_multivalue(cell) for cell in cells.dropna().unique()}
    pairs = pd.DataFrame({"act_id": df["source_id"], "value": cells.map(split_cache)})
    pairs = pairs.explode("value").dropna(subset=["value"])
    return pairs.drop_duplicates().reset_index(drop=True)


def build_many_to_many(df: pd.DataFrame, column: str, id_column: str) -> tuple:
    """Return (lookup table, link table) for a multi-valued column."""
    pairs = explode_multivalue(df, column)
    lookup = build_lookup(pairs["value"], id_column)
    pairs[id_column] = to_ids(pairs["value"], lookup, id_column)
    return lookup, pairs[["act_id", id_column]].drop_duplicates()


def build_bodies(df: pd.DataFrame) -> tuple:
    """Bodies need one more step: the note in parentheses is moved to its own column."""
    pairs = explode_multivalue(df, "adopting_body")
    parsed = {value: split_body(value) for value in pairs["value"].unique()}
    pairs["name"] = pairs["value"].map(lambda v: parsed[v][0])
    pairs["history_note"] = pairs["value"].map(lambda v: parsed[v][1])

    bodies = build_lookup(pairs["name"], "body_id")
    notes = pairs.dropna(subset=["history_note"]).groupby("name")["history_note"].first()
    bodies["history_note"] = bodies["name"].map(notes)
    pairs["body_id"] = to_ids(pairs["name"], bodies, "body_id")
    return bodies, pairs[["act_id", "body_id"]].drop_duplicates()


def normalize(df: pd.DataFrame) -> dict:
    """Split the flat DataFrame into the relational tables (dict name -> DataFrame)."""
    df = df.copy()
    # a few rows have a numeric code ('3700') instead of a section name -> unknown
    section = df["database_section"].astype("object")
    df["database_section"] = section.where(~section.str.fullmatch(r"\d+", na=False))

    tables = {
        "database_sections": build_lookup(df["database_section"], "section_id"),
        "legal_forces": build_lookup(df["legal_force"], "legal_force_id"),
        "regions": build_lookup(df["region"], "region_id"),
    }
    tables["areas"], tables["act_areas"] = build_many_to_many(df, "legal_relation_area", "area_id")
    tables["forms"], tables["act_forms"] = build_many_to_many(df, "act_form", "form_id")
    tables["bodies"], tables["act_bodies"] = build_bodies(df)

    acts = pd.DataFrame({
        "act_id": df["source_id"],
        "law_id": df["law_id"],
        "title": df["title"],
        "is_repealed": df["is_repealed"].astype(int),
        "section_id": to_ids(df["database_section"], tables["database_sections"], "section_id"),
        "legal_force_id": to_ids(df["legal_force"], tables["legal_forces"], "legal_force_id"),
        "region_id": to_ids(df["region"], tables["regions"], "region_id"),
    })
    for col in ["date_adopted", "date_modified", "date_published"]:
        text = df[col].dt.strftime("%Y-%m-%d")
        acts[col] = text.astype("object").where(df[col].notna(), None)
    tables["acts"] = acts
    return tables


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------
def build_database(parquet_path: Path, db_path: Path) -> dict:
    """Create the SQLite file from the Parquet file. Returns a small report."""
    db_path = Path(db_path)
    with Timer() as timer:
        df = load_acts(parquet_path, columns=SOURCE_COLUMNS, optimize=False)
        tables = normalize(df)

        if db_path.exists():
            db_path.unlink()                      # always rebuild from scratch
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        try:
            connection.executescript(SCHEMA_SQL)
            for name in TABLE_ORDER:              # parents first, then children
                tables[name].to_sql(name, connection, if_exists="append", index=False)
            connection.commit()
            broken_links = connection.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            connection.close()

    return {
        "db_path": config.project_path(db_path),
        "db_file_bytes": db_path.stat().st_size,
        "seconds": round(timer.seconds, 2),
        "rows": {name: len(tables[name]) for name in TABLE_ORDER},
        "foreign_key_violations": len(broken_links),
    }


def run_query(db_path: Path, sql: str, params: dict = None) -> pd.DataFrame:
    """Run a SELECT and return the result as a pandas DataFrame."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Build the database first.")
    connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, connection, params=params or {})
    finally:
        connection.close()


def get_db_path(paths: dict) -> Path:
    """SQLite file that belongs to the full or the sample data."""
    return config.PROCESSED_DIR / f"acts_{paths['name']}.db"
