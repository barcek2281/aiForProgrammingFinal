"""Project-wide settings: paths, dataset location and column groups.

Every other module imports its constants from here, so a path or a column
name has to be changed in one place only.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REPORTS_DIR = BASE_DIR / "reports"

# --------------------------------------------------------------------------
# Dataset (Hugging Face)
# --------------------------------------------------------------------------
HF_REPO_ID = "Arailym-tleubayeva/KazakhLawCorpus-clean"
HF_FILENAME = "data/kazakhlawcorpus_kk_only.csv"
RAW_CSV = RAW_DIR / HF_FILENAME
SAMPLE_CSV = SAMPLE_DIR / "acts_sample_500.csv"

# Rows per chunk when the big CSV is streamed.
CHUNK_SIZE = 50_000

# --------------------------------------------------------------------------
# Column groups (used for type conversion and for the audit)
# --------------------------------------------------------------------------
ID_COLUMN = "law_id"
TEXT_COLUMN = "title"

DATE_COLUMNS = [
    "date_adopted",
    "date_modified",
    "date_published",
    "ministry_registration_date",
]
DATE_FORMAT = "%Y-%m-%d"

# Stored as 't' / 'f' in the CSV (PostgreSQL export style).
BOOL_COLUMNS = ["is_archive", "is_repealed"]
BOOL_MAP = {"t": True, "f": False}

INT_COLUMNS = ["source_id"]
FLOAT_COLUMNS = ["state_registry_number"]

# Low-cardinality text columns -> pandas 'category' after loading.
CATEGORY_COLUMNS = [
    "act_form",
    "legal_force",
    "adopting_body",
    "regulatory_body",
    "developer_body",
    "region",
    "place_of_adoption",
    "database_section",
    "legal_relation_area",
]

# Columns that hold several values in one cell.
MULTIVALUE_COLUMNS = ["legal_relation_area", "act_form", "adopting_body"]

# Strings that mean "no value". Compared case-insensitively after strip().
# 'ЖОҚ' is Kazakh and 'НЕТ' is Russian for "no / none".
PLACEHOLDERS = ["", "null", "none", "nan", "undefined", "жоқ", "нет", "n/a", "-", "--", "---"]


def get_paths(use_sample: bool = False) -> dict:
    """Return the csv / parquet / reports paths for the full or the sample data."""
    if use_sample:
        name = "sample"
        csv_path = SAMPLE_CSV
    else:
        name = "full"
        csv_path = RAW_CSV
    return {
        "name": name,
        "csv": csv_path,
        "parquet": PROCESSED_DIR / f"acts_{name}.parquet",
        "reports": REPORTS_DIR / name,
    }
