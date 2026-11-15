"""Loading the corpus: download, chunked CSV -> Parquet, dtype optimisation.

The raw CSV is ~337 MB, so it is never loaded in one piece during conversion.
It is streamed in chunks, every chunk is cleaned and appended to one Parquet
file. Memory usage is measured before and after each step.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.utils import Timer, format_bytes, now_str

try:  # psutil is only needed for the peak-memory measurement
    import psutil
except ImportError:
    psutil = None


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download the CSV from Hugging Face into data/raw (skipped if present)."""
    if config.RAW_CSV.exists() and not force:
        print(f"Already downloaded: {config.RAW_CSV}")
        return config.RAW_CSV

    from huggingface_hub import hf_hub_download  # imported late: optional dependency

    print(f"Downloading {config.HF_FILENAME} from {config.HF_REPO_ID} (~337 MB)...")
    path = hf_hub_download(
        repo_id=config.HF_REPO_ID,
        filename=config.HF_FILENAME,
        repo_type="dataset",
        local_dir=config.RAW_DIR,
    )
    print(f"Saved to {path}")
    return Path(path)


# --------------------------------------------------------------------------
# Memory helpers
# --------------------------------------------------------------------------
def memory_bytes(df: pd.DataFrame) -> int:
    """Real memory used by a DataFrame (deep=True also counts string contents)."""
    return int(df.memory_usage(deep=True).sum())


def process_memory_bytes() -> int:
    """Resident memory of the current Python process (0 if psutil is missing)."""
    if psutil is None:
        return 0
    return int(psutil.Process().memory_info().rss)


# --------------------------------------------------------------------------
# Cleaning of one chunk
# --------------------------------------------------------------------------
def count_placeholders(chunk: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """Find placeholder strings ('ЖОҚ', 'undefined', '' ...) in every column.

    `counts` is updated in place: counts[column][placeholder] += n.
    Returns a boolean DataFrame (True = the cell is a placeholder).
    """
    mask = pd.DataFrame(False, index=chunk.index, columns=chunk.columns)
    for col in chunk.columns:
        stripped = chunk[col].str.strip()
        is_placeholder = stripped.str.lower().isin(config.PLACEHOLDERS)
        mask[col] = is_placeholder
        for token, n in stripped[is_placeholder].value_counts().items():
            key = token if token != "" else "<empty>"
            counts[col][key] += int(n)
    return mask


def clean_chunk(chunk: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Clean one raw chunk (all columns are strings) and set proper dtypes.

    Steps: strip spaces -> placeholders to NA -> dates -> booleans -> numbers.
    `stats` collects audit information across chunks (placeholders, date
    patterns, values that could not be parsed).
    """
    placeholder_mask = count_placeholders(chunk, stats["placeholders"])
    clean = pd.DataFrame(index=chunk.index)

    for col in chunk.columns:
        values = chunk[col].str.strip().mask(placeholder_mask[col])

        if col in config.DATE_COLUMNS:
            # remember the raw "shape" of the value, e.g. 2004-07-15 -> dddd-dd-dd
            patterns = values.dropna().str.replace(r"\d", "d", regex=True).value_counts()
            for pattern, n in patterns.items():
                stats["date_patterns"][col][pattern] += int(n)
            parsed = pd.to_datetime(values, format=config.DATE_FORMAT, errors="coerce")
            stats["unparsed"][col] += int((values.notna() & parsed.isna()).sum())
            clean[col] = parsed
        elif col in config.BOOL_COLUMNS:
            mapped = values.map(config.BOOL_MAP)
            stats["unparsed"][col] += int((values.notna() & mapped.isna()).sum())
            clean[col] = mapped.astype("boolean")
        elif col in config.INT_COLUMNS or col in config.FLOAT_COLUMNS:
            numbers = pd.to_numeric(values, errors="coerce")
            stats["unparsed"][col] += int((values.notna() & numbers.isna()).sum())
            clean[col] = numbers.astype("Int64") if col in config.INT_COLUMNS else numbers
        else:
            clean[col] = values
    return clean


def build_arrow_schema(columns: list) -> pa.Schema:
    """Fixed Parquet schema, so that every chunk is written with the same types."""
    fields = []
    for col in columns:
        if col in config.DATE_COLUMNS:
            arrow_type = pa.timestamp("ms")
        elif col in config.BOOL_COLUMNS:
            arrow_type = pa.bool_()
        elif col in config.INT_COLUMNS:
            arrow_type = pa.int64()
        elif col in config.FLOAT_COLUMNS:
            arrow_type = pa.float64()
        else:
            arrow_type = pa.string()
        fields.append(pa.field(col, arrow_type))
    return pa.schema(fields)


# --------------------------------------------------------------------------
# CSV -> Parquet
# --------------------------------------------------------------------------
def convert_csv_to_parquet(csv_path: Path, parquet_path: Path,
                           chunksize: int = config.CHUNK_SIZE) -> dict:
    """Stream the CSV in chunks, clean every chunk and append it to Parquet.

    Returns a report (rows, sizes, memory, time, placeholder statistics).
    """
    csv_path, parquet_path = Path(csv_path), Path(parquet_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "placeholders": defaultdict(lambda: defaultdict(int)),
        "date_patterns": defaultdict(lambda: defaultdict(int)),
        "unparsed": defaultdict(int),
    }
    n_rows = n_chunks = 0
    raw_memory = clean_memory = peak_chunk_memory = 0
    peak_process_memory = process_memory_bytes()
    writer = schema = None

    with Timer() as timer:
        # dtype=str + na_filter=False: read every cell exactly as written,
        # otherwise pandas would silently hide placeholders such as 'null'.
        reader = pd.read_csv(csv_path, chunksize=chunksize, dtype=str,
                             na_filter=False, encoding="utf-8")
        try:
            for chunk in reader:
                chunk_memory = memory_bytes(chunk)
                raw_memory += chunk_memory
                peak_chunk_memory = max(peak_chunk_memory, chunk_memory)

                clean = clean_chunk(chunk, stats)
                clean_memory += memory_bytes(clean)

                if writer is None:
                    schema = build_arrow_schema(list(clean.columns))
                    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
                table = pa.Table.from_pandas(clean, schema=schema, preserve_index=False)
                writer.write_table(table)

                n_rows += len(chunk)
                n_chunks += 1
                peak_process_memory = max(peak_process_memory, process_memory_bytes())
                print(f"  chunk {n_chunks}: {n_rows:,} rows written")
        finally:
            if writer is not None:
                writer.close()

    if n_rows == 0:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    missing_columns = [c for c in [config.ID_COLUMN, config.TEXT_COLUMN] + config.DATE_COLUMNS
                       if c not in schema.names]
    return {
        "created_at": now_str(),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "rows": n_rows,
        "columns": len(schema.names),
        "chunks": n_chunks,
        "chunksize": chunksize,
        "seconds": round(timer.seconds, 2),
        "csv_file_bytes": csv_path.stat().st_size,
        "parquet_file_bytes": parquet_path.stat().st_size,
        "memory_raw_strings_bytes": raw_memory,
        "memory_clean_typed_bytes": clean_memory,
        "peak_chunk_memory_bytes": peak_chunk_memory,
        "peak_process_memory_bytes": peak_process_memory,
        "expected_columns_missing": missing_columns,
        "placeholders": {col: dict(tokens) for col, tokens in stats["placeholders"].items()},
        "date_patterns": {col: dict(p) for col, p in stats["date_patterns"].items()},
        "unparsed_values": dict(stats["unparsed"]),
    }


def print_conversion_report(report: dict) -> None:
    """Show the main numbers of the CSV -> Parquet conversion."""
    csv_size, pq_size = report["csv_file_bytes"], report["parquet_file_bytes"]
    raw_mem, clean_mem = report["memory_raw_strings_bytes"], report["memory_clean_typed_bytes"]
    print(f"Rows: {report['rows']:,} | columns: {report['columns']} | "
          f"chunks: {report['chunks']} x {report['chunksize']:,} | time: {report['seconds']} s")
    print(f"File size : CSV {format_bytes(csv_size)} -> Parquet {format_bytes(pq_size)} "
          f"({csv_size / pq_size:.1f}x smaller)")
    print(f"In memory : raw strings {format_bytes(raw_mem)} -> cleaned + typed {format_bytes(clean_mem)} "
          f"({100 * (1 - clean_mem / raw_mem):.1f}% saved)")
    print(f"Peak      : largest chunk {format_bytes(report['peak_chunk_memory_bytes'])}, "
          f"Python process {format_bytes(report['peak_process_memory_bytes'])}")
    if report["expected_columns_missing"]:
        print(f"WARNING: expected columns not found: {report['expected_columns_missing']}")


# --------------------------------------------------------------------------
# Loading the Parquet file
# --------------------------------------------------------------------------
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink a DataFrame: repeated strings -> category, big ints -> small ints."""
    df = df.copy()
    for col in df.columns:
        if col in config.CATEGORY_COLUMNS:
            df[col] = df[col].astype("category")
        elif col in config.INT_COLUMNS:
            if df[col].isna().any():
                df[col] = df[col].astype("Int32")
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col in config.FLOAT_COLUMNS:
            present = df[col].dropna()
            if len(present) > 0 and (present % 1 == 0).all():   # whole numbers stored as float
                df[col] = df[col].astype("Int32")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_acts(parquet_path: Path, columns: list = None, optimize: bool = True) -> pd.DataFrame:
    """Read the Parquet file (optionally only some columns) into pandas."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}. Run the conversion first.")
    df = pd.read_parquet(parquet_path, columns=columns)
    if optimize:
        df = optimize_dtypes(df)
    return df


def dtype_memory_table(parquet_path: Path) -> pd.DataFrame:
    """Per-column memory before / after dtype optimisation."""
    before = load_acts(parquet_path, optimize=False)
    after = optimize_dtypes(before)
    mem_before = before.memory_usage(deep=True).drop("Index")
    mem_after = after.memory_usage(deep=True).drop("Index")
    table = pd.DataFrame({
        "dtype_before": before.dtypes.astype(str),
        "dtype_after": after.dtypes.astype(str),
        "mb_before": (mem_before / 1024 ** 2).round(2),
        "mb_after": (mem_after / 1024 ** 2).round(2),
    })
    table["saved_pct"] = (100 * (1 - table["mb_after"] / table["mb_before"].where(table["mb_before"] > 0))).round(1)
    return table.sort_values("mb_before", ascending=False)


# --------------------------------------------------------------------------
# Benchmark: CSV vs Parquet
# --------------------------------------------------------------------------
def benchmark_formats(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """Compare loading strategies by time, memory and file size."""
    model_columns = [config.TEXT_COLUMN, "database_section", "date_adopted"]
    strategies = [
        ("CSV, pandas defaults (baseline)", csv_path,
         lambda: pd.read_csv(csv_path, low_memory=False)),
        ("Parquet, no optimisation", parquet_path,
         lambda: load_acts(parquet_path, optimize=False)),
        ("Parquet + optimised dtypes", parquet_path,
         lambda: load_acts(parquet_path, optimize=True)),
        ("Parquet, 3 columns only", parquet_path,
         lambda: load_acts(parquet_path, columns=model_columns, optimize=True)),
    ]
    rows = []
    for name, path, load in strategies:
        with Timer() as timer:
            df = load()
        rows.append({
            "strategy": name,
            "file_mb": round(Path(path).stat().st_size / 1024 ** 2, 2),
            "load_seconds": round(timer.seconds, 3),
            "memory_mb": round(memory_bytes(df) / 1024 ** 2, 2),
            "rows": len(df),
            "columns": df.shape[1],
        })
        del df
    return pd.DataFrame(rows)
