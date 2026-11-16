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
| Kazakh text preprocessing: tokens, stop words, suffix stemmer | `src/preprocess.py` | done |
| Train/test split without leakage | `src/dataset.py` | done |
| Baselines, three TF-IDF models and LaBSE embeddings | `src/models.py`, `src/embeddings.py` | done |
| Error analysis | `src/error_analysis.py` | done |
| Prediction for a title typed by the user | `main.py`, item 12 | done |

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
(SQL analytics is item **8**, EDA is item **9**). Item **10** trains the models (about 10 minutes
without LaBSE encoding), item **11** analyses the errors and item **12** classifies a title you type.

To try the program in a few seconds without the download, use the 500-row sample
that is stored in the repository:

```bash
python main.py --sample
```

Every step can also be run without the menu:

```bash
python main.py --sample --action all      # convert -> memory -> benchmark -> audit -> ... -> db -> sql -> eda -> train -> errors
python main.py --action convert           # full data, one step
```

### Tests

The logic that is written by hand - the Kazakh suffix stemmer, the splitter for multi-valued
cells, title normalisation and the leakage-free split - is covered by unit tests. They use only
the standard library's `unittest`, need no dataset and run in under a second:

```bash
python -m unittest                        # all tests
python -m unittest tests.test_preprocess -v
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
src/preprocess.py       Kazakh tokeniser, stop words, suffix stemmer
src/dataset.py          labelled data from SQLite, time-based split, validation split
src/models.py           baselines, TF-IDF models, LaBSE model, comparison, prediction
src/embeddings.py       LaBSE sentence embeddings with a disk cache
src/error_analysis.py   per-class metrics, confusions, bootstrap intervals, charts
models/                 trained classifier for user input (not in git)
docs/er_diagram.md      ER diagram and normalisation decisions
src/utils.py            timer, byte formatting, JSON helpers
scripts/fetch_sample.py rebuilds data/sample/acts_sample_500.csv from the dataset-viewer API
tests/                  unit tests for preprocessing, audit, database helpers and the split
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

## Models

**Task.** Predict `database_section` (35 thematic sections) from the Kazakh title only.
`description` is never used: it contains the act form and number in plain text.

**Split without leakage** (`src/dataset.py`):

| Part | Rule | Acts |
|---|---|---|
| Train | adopted before 2023-01-01 | 171,094 |
| Validation (inside train) | the last 12 months of the training period, for choosing parameters | 12,910 |
| Test | adopted from 2023-01-01, titles that already occur in train removed (5,924 acts) | 38,612 |

**Kazakh morphology** (`src/preprocess.py`). Three representations are compared: word forms,
stems from a rule-based suffix stemmer (case, possessive, plural and word-forming endings are cut
in that order, vowel harmony is checked), and character n-grams. Stemming reduces the training
vocabulary from 31,981 to 20,064 words and the share of unseen test words from 0.49% to 0.33%.

**Results on the test set** (`reports/full/models/model_comparison.csv`):

| Model | F1-macro | 95% interval | Accuracy | Features | Feature time | Fit time | Train matrix |
|---|---|---|---|---|---|---|---|
| Baseline: always the largest class | 0.017 | — | 0.439 | — | — | 0.1 s | — |
| Baseline: random by class frequency | 0.026 | — | 0.172 | — | — | 0.1 s | — |
| Word forms TF-IDF + LinearSVC | **0.590** | 0.576–0.614 | 0.793 | 119,153 | 2.4 s | 14.0 s | 33.9 MB |
| Stems TF-IDF + LinearSVC | **0.590** | 0.576–0.614 | 0.797 | 89,609 | 2.5 s | 9.0 s | 32.6 MB |
| Char n-grams (2–4) TF-IDF + LinearSVC | 0.575 | 0.562–0.601 | 0.792 | 51,293 | 19.0 s | 33.6 s | 437.1 MB |
| LaBSE embeddings + LogisticRegression | 0.533 | 0.517–0.556 | 0.764 | 768 | 1,386 s | 58.7 s | 501.3 MB |

Intervals come from 200 bootstrap resamples of the test set.

**Conclusions.**

- All four models are far above both baselines.
- The three TF-IDF models are statistically tied: their intervals overlap. Stemming gives the same
  quality as word forms with 25% fewer features and the fastest training.
- LaBSE is clearly below TF-IDF (its interval does not overlap with word forms or stems) and costs
  23 minutes of encoding on an Apple-silicon GPU. Legal titles are classified by specific terms,
  which sparse features keep exactly and a general-purpose sentence vector blurs.
- The classifier is linear in all cases: LinearSVC for sparse features, logistic regression for
  embeddings, because the SVM solver needs more than five minutes per fit on the dense matrix.

**Error analysis** (`reports/full/error_analysis/`, model: word forms TF-IDF).

- 27,284 of 38,612 test acts are classified correctly by all four models; 5,664 are missed by all
  four, so most errors come from the data and not from the representation.
- The most frequent confusion: 20.7% of "Конституциялық құрылыс және мемлекеттік басқару
  негіздері" acts are predicted as "Қаржы".
- Titles of 1–5 words have a 50.8% error rate, titles of 6–10 words 11.9%.
- Sections with fewer than 50 test acts have a mean F1 of 0.272 against 0.669 for the larger
  ones; two of them have F1 = 0 (1 and 8 test acts).

**Limitations.** Labels are noisy (4,276 titles occur with more than one section). The class mix
shifts over time. The stemmer has no dictionary, so it sometimes cuts too much
(`облысы` → `облы`). LaBSE is used frozen, without fine-tuning.

**LaBSE is optional.** Without `sentence-transformers` the comparison runs with the TF-IDF models
only. Embeddings are cached in `data/processed/`, so the encoding is done once.
