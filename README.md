# Legal Act Classification by Regulation Area

Final project for *Programming for Data Analysis and Database* (Applied AI, group AAI-2603M).
Team: Zhunis Ayanat, Kossanov Adilkhan, Aitzhan Adilet.

**Goal:** predict the area of legal regulation of a Kazakh legal act from its title, comparing
word TF-IDF, character n-gram TF-IDF and LaBSE embeddings (metric: F1-macro).

**Data:** [KazakhLawCorpus-clean](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean)
— 215,889 acts, 25 columns, one 337 MB CSV file, licence CC BY 4.0.

## Project status

| Stage | Module | Status |
|---|---|---|
| Chunked loading, CSV -> Parquet, dtype optimisation with memory measurements | `src/data_loader.py` | done |
| Metadata audit (missing values, placeholders, dates, multi-valued fields, duplicates) | `src/audit.py` | done |
| Relational schema in SQLite (ER diagram: [docs/er_diagram.md](docs/er_diagram.md)) | `src/db.py` | done |
| SQL analytics: 12 queries with JOIN, HAVING, CTE and window functions | `src/queries.py` | done |
| EDA: 7 charts, one interactive Plotly chart, findings tied to numbers | `src/eda.py` | done |
| Kazakh text preprocessing, models, error analysis, prediction for user input | — | to do |

## How to run

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The menu starts with the full dataset: choose **1** to download the CSV (337 MB, once),
then **2** to convert it to Parquet and **7** to build the SQLite database, then any other item
(SQL analytics is item **8**, EDA is item **9**).

To try the program in a few seconds without the download, use the 500-row sample
that is stored in the repository:

```bash
python main.py --sample
```

Every step can also be run without the menu:

```bash
python main.py --sample --action all      # convert -> memory -> benchmark -> audit -> overview -> db -> sql -> eda
python main.py --action convert           # full data, one step
```

## Project structure

```
main.py                 console menu and command-line options
src/config.py           paths, dataset location, column groups, placeholder list
src/data_loader.py      download, chunked CSV -> Parquet, dtype optimisation, benchmark
src/audit.py            metadata audit
src/db.py               relational schema, normalisation, loading into SQLite
src/queries.py          analytical SQL queries (results go to pandas and CSV)
src/eda.py              charts and findings
docs/er_diagram.md      ER diagram and normalisation decisions
src/utils.py            timer, byte formatting, JSON helpers
scripts/fetch_sample.py rebuilds data/sample/acts_sample_500.csv from the dataset-viewer API
data/raw/               downloaded CSV (not in git)
data/processed/         Parquet files and SQLite databases (not in git)
data/sample/            500-row sample (in git)
reports/full/           results for the full dataset: JSON and CSV tables, charts in eda/
```

## What the data stage does

1. **Chunked reading.** The CSV is read 50,000 rows at a time with every cell kept as a raw
   string, so placeholder values are counted before they are replaced.
2. **Cleaning.** Placeholders (`ЖОҚ`, `НЕТ`, `undefined`, empty cells, ...) become missing
   values; dates, booleans (`t`/`f`) and numbers get proper types.
3. **Parquet.** Chunks are appended to one zstd-compressed Parquet file with a fixed schema.
4. **Dtype optimisation.** Repeated strings become `category`, integers are downcast.
   Memory is measured with `memory_usage(deep=True)` before and after.
5. **Audit.** Results are printed and saved to `reports/<full|sample>/audit/`.

Multi-valued fields (`legal_relation_area`, `act_form`, `adopting_body`) cannot be split with a
plain `split(",")`, because many values contain commas themselves. `audit.split_multivalue`
splits only on a comma that is outside parentheses and is followed by an upper-case letter.

## Database and SQL

`src/db.py` splits the flat table into 10 tables: `acts`, three lookup tables and three
many-to-many relations (areas, forms and adopting bodies). The schema is created with explicit
`CREATE TABLE` statements (primary and foreign keys, indexes) and checked with
`PRAGMA foreign_key_check` after loading.

`src/queries.py` holds 12 queries. Each result is loaded into pandas and saved to
`reports/<full|sample>/sql/`. Queries 11 and 12 define the time-based train/test split and
measure how many test titles already occur in the training period.

## EDA

`src/eda.py` (menu item 9) saves seven PNG charts and one interactive Plotly chart to
`reports/<full|sample>/eda/`, and writes the conclusions with their numbers to
`eda_findings.md`. The interactive HTML file is not stored in git (it embeds plotly.js);
run item 9 to create it.

| Chart | What it shows |
|---|---|
| `01_class_distribution.png` | 35 target classes, from 64,854 acts down to 110 |
| `02_acts_per_year.png` | acts per year, 3-year moving average, start of the test period |
| `03_title_length.png` | title length in words, median and IQR outlier limit |
| `04_top_words.png` | the 20 most frequent words in titles |
| `05_section_by_period.png` | share of each section in every five-year period |
| `06_train_test_shift.png` | class shares before and after the split date |
| `07_top_areas.png` | most frequent areas of legal regulation |
| `08_sections_per_year_interactive.html` | acts per year for the six largest sections |
