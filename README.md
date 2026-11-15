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
| Relational schema, SQLite, SQL analytics | — | to do |
| EDA and visualisations | — | to do |
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
then **2** to convert it to Parquet, then any other item.

To try the program in a few seconds without the download, use the 500-row sample
that is stored in the repository:

```bash
python main.py --sample
```

Every step can also be run without the menu:

```bash
python main.py --sample --action all      # convert -> memory -> benchmark -> audit -> overview
python main.py --action convert           # full data, one step
```

## Project structure

```
main.py                 console menu and command-line options
src/config.py           paths, dataset location, column groups, placeholder list
src/data_loader.py      download, chunked CSV -> Parquet, dtype optimisation, benchmark
src/audit.py            metadata audit
src/utils.py            timer, byte formatting, JSON helpers
scripts/fetch_sample.py rebuilds data/sample/acts_sample_500.csv from the dataset-viewer API
data/raw/               downloaded CSV (not in git)
data/processed/         Parquet files (not in git)
data/sample/            500-row sample (in git)
reports/full/           results for the full dataset: JSON and CSV tables
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
