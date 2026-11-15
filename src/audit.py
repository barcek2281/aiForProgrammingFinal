"""Metadata audit: missing values, placeholders, dates, multi-valued fields,
duplicates and constant columns.

Every function returns a DataFrame (or a dict of DataFrames), so the same
results can be printed in the console, saved to CSV and used in the report.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.utils import print_section, save_json


# --------------------------------------------------------------------------
# 1. Missing values
# --------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values, unique values and dtype for every column."""
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (100 * df.isna().mean()).round(2),
        "unique": df.nunique(),
    })
    return report.sort_values("missing_pct", ascending=False)


# --------------------------------------------------------------------------
# 2. Placeholder values (collected while the CSV was converted)
# --------------------------------------------------------------------------
def placeholder_report(conversion_report: dict) -> pd.DataFrame:
    """Table 'column x placeholder' with the number of cells replaced by NA."""
    rows = []
    n_rows = conversion_report["rows"]
    for col, tokens in conversion_report["placeholders"].items():
        for token, count in tokens.items():
            rows.append({
                "column": col,
                "placeholder": token,
                "count": count,
                "pct_of_rows": round(100 * count / n_rows, 2),
            })
    if not rows:
        return pd.DataFrame(columns=["column", "placeholder", "count", "pct_of_rows"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Dates
# --------------------------------------------------------------------------
def date_report(df: pd.DataFrame, conversion_report: dict) -> pd.DataFrame:
    """Range, raw formats and suspicious values for every date column."""
    today = pd.Timestamp(datetime.now().date())
    rows = []
    for col in config.DATE_COLUMNS:
        if col not in df.columns:
            continue
        dates = df[col].dropna()
        patterns = conversion_report["date_patterns"].get(col, {})
        row = {
            "column": col,
            "present": len(dates),
            "missing_pct": round(100 * df[col].isna().mean(), 2),
            "min": dates.min().date() if len(dates) else None,
            "max": dates.max().date() if len(dates) else None,
            "in_future": int((dates > today).sum()),
            "unparsed": conversion_report["unparsed_values"].get(col, 0),
            "raw_formats": ", ".join(f"{p} ({n})" for p, n in patterns.items()),
        }
        if col != "date_adopted" and "date_adopted" in df.columns:
            # an act cannot be modified / published before it was adopted
            row["before_adoption"] = int((df[col] < df["date_adopted"]).sum())
        else:
            row["before_adoption"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Multi-valued fields
# --------------------------------------------------------------------------
def split_multivalue(value) -> list:
    """Split a cell such as 'Су көлігі, Көліктегі қауіпсіздік' into values.

    A plain value.split(',') is wrong here, because many values contain commas
    themselves ('Мал шаруашылығы, аң шаруашылығы, құс шаруашылығы'). Rule used:
    a comma separates two values only if it is outside parentheses AND the
    next word starts with an upper-case letter.
    """
    if value is None or pd.isna(value):
        return []
    text = str(value)
    parts = []
    depth = 0      # how many '(' are currently open
    start = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            rest = text[i + 1:].lstrip()
            if rest and rest[0].isupper():
                parts.append(text[start:i].strip())
                start = i + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def multivalue_report(df: pd.DataFrame, column: str, top_n: int = 15) -> dict:
    """Statistics of one multi-valued column.

    Returns {'summary': dict, 'top_values': DataFrame, 'values_per_row': DataFrame}.
    """
    cells = df[column].dropna().astype(str)
    # split every distinct cell only once (cells repeat a lot)
    split_cache = {cell: split_multivalue(cell) for cell in cells.unique()}
    lists = cells.map(split_cache)
    n_values = lists.map(len)

    value_counts = Counter()
    naive_values = set()
    for cell, count in cells.value_counts().items():
        for value in split_cache[cell]:
            value_counts[value] += count
        naive_values.update(part.strip() for part in cell.split(","))

    # 'Шешім, Өзгерту' and 'Өзгерту, Шешім' are the same set in a different order
    distinct_sets = {tuple(sorted(values)) for values in split_cache.values()}

    summary = {
        "column": column,
        "non_null_rows": len(cells),
        "multi_valued_rows": int((n_values > 1).sum()),
        "multi_valued_pct": round(100 * (n_values > 1).mean(), 2) if len(cells) else 0.0,
        "mean_values_per_row": round(float(n_values.mean()), 2) if len(cells) else 0.0,
        "max_values_per_row": int(n_values.max()) if len(cells) else 0,
        "distinct_raw_cells": len(split_cache),
        "distinct_cells_ignoring_order": len(distinct_sets),
        "distinct_values": len(value_counts),
        "distinct_values_naive_comma_split": len(naive_values),
        "values_seen_once": sum(1 for n in value_counts.values() if n == 1),
    }
    top_values = pd.DataFrame(value_counts.most_common(top_n), columns=["value", "rows"])
    top_values["pct_of_rows"] = (100 * top_values["rows"] / max(len(cells), 1)).round(2)
    values_per_row = (n_values.value_counts().sort_index()
                      .rename_axis("values_in_cell").reset_index(name="rows"))
    return {"summary": summary, "top_values": top_values, "values_per_row": values_per_row}


# --------------------------------------------------------------------------
# 5. Duplicates
# --------------------------------------------------------------------------
def normalize_title(titles: pd.Series) -> pd.Series:
    """Lower-case, remove quotes and repeated spaces - to find near-identical titles."""
    return (titles.astype(str).str.lower()
            .str.replace(r"[\"«»“”„']", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def duplicate_report(df: pd.DataFrame, label_column: str = "database_section") -> dict:
    """Duplicate ids and titles. Title duplicates matter for the train/test split."""
    summary = {"rows": len(df)}
    top_titles = pd.DataFrame(columns=["title", "rows"])

    if config.ID_COLUMN in df.columns:
        summary["duplicate_ids"] = int(df[config.ID_COLUMN].duplicated().sum())
    summary["fully_duplicated_rows"] = int(df.duplicated().sum())

    if config.TEXT_COLUMN in df.columns:
        titles = df[config.TEXT_COLUMN]
        normalized = normalize_title(titles)
        summary["missing_titles"] = int(titles.isna().sum())
        summary["distinct_titles"] = int(titles.nunique())
        summary["rows_with_repeated_title"] = int(titles.duplicated(keep=False).sum())
        summary["rows_with_repeated_title_pct"] = round(100 * titles.duplicated(keep=False).mean(), 2)
        summary["distinct_titles_normalized"] = int(normalized.nunique())
        summary["rows_with_repeated_title_normalized"] = int(normalized.duplicated(keep=False).sum())

        if label_column in df.columns:
            # the same title with different labels = noise that no model can learn
            labels_per_title = df.groupby(normalized, observed=True)[label_column].nunique()
            summary[f"titles_with_conflicting_{label_column}"] = int((labels_per_title > 1).sum())

        top_titles = (titles.value_counts().head(10)
                      .rename_axis("title").reset_index(name="rows"))
    return {"summary": summary, "top_titles": top_titles}


# --------------------------------------------------------------------------
# 6. Constant / almost constant columns
# --------------------------------------------------------------------------
def constant_columns_report(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Columns where one value (or NA) covers at least `threshold` of the rows."""
    rows = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        if len(counts) == 0:
            continue
        top_share = counts.iloc[0] / len(df)
        if df[col].nunique(dropna=False) <= 1:
            verdict = "constant"
        elif top_share >= threshold:
            verdict = "almost constant"
        else:
            continue
        rows.append({
            "column": col,
            "verdict": verdict,
            "top_value": str(counts.index[0])[:60],
            "top_share_pct": round(100 * top_share, 2),
        })
    return pd.DataFrame(rows, columns=["column", "verdict", "top_value", "top_share_pct"])


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------
def run_full_audit(df: pd.DataFrame, conversion_report: dict, out_dir: Path) -> dict:
    """Run all checks, print them and save the tables to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}       # file name -> DataFrame
    summary = {"rows": len(df), "columns": df.shape[1]}

    print_section("1. Missing values by column")
    tables["missing"] = missing_report(df).rename_axis("column").reset_index()
    print(tables["missing"].to_string(index=False))

    print_section("2. Placeholder values replaced by NA")
    tables["placeholders"] = placeholder_report(conversion_report)
    if tables["placeholders"].empty:
        print("No placeholder values found.")
    else:
        print(tables["placeholders"].to_string(index=False))

    print_section("3. Date columns")
    tables["dates"] = date_report(df, conversion_report)
    print(tables["dates"].to_string(index=False))

    print_section("4. Multi-valued fields")
    summary["multivalue"] = {}
    for col in config.MULTIVALUE_COLUMNS:
        if col not in df.columns:
            continue
        result = multivalue_report(df, col)
        summary["multivalue"][col] = result["summary"]
        tables[f"multivalue_{col}_top"] = result["top_values"]
        tables[f"multivalue_{col}_sizes"] = result["values_per_row"]
        print(f"\n--- {col} ---")
        for key, value in result["summary"].items():
            if key != "column":
                print(f"  {key:36s} {value}")
        print(result["top_values"].head(8).to_string(index=False))

    print_section("5. Duplicates")
    duplicates = duplicate_report(df)
    summary["duplicates"] = duplicates["summary"]
    tables["duplicate_titles_top"] = duplicates["top_titles"]
    for key, value in duplicates["summary"].items():
        print(f"  {key:44s} {value}")
    print("\nMost repeated titles:")
    for _, row in duplicates["top_titles"].head(5).iterrows():
        print(f"  {row['rows']:>6}  {str(row['title'])[:95]}")

    print_section("6. Constant and almost constant columns")
    tables["constant_columns"] = constant_columns_report(df)
    if tables["constant_columns"].empty:
        print("None.")
    else:
        print(tables["constant_columns"].to_string(index=False))

    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    save_json(summary, out_dir / "audit_summary.json")
    print(f"\nSaved {len(tables)} tables and audit_summary.json to {out_dir}")
    return summary
