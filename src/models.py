"""Models: baselines and four text representations with the same classifier.

Only the representation of the title changes between the models:

    word_tfidf   TF-IDF of word forms (unigrams + bigrams)
    stem_tfidf   TF-IDF of stems after suffix stemming
    char_tfidf   TF-IDF of character n-grams (2-4 characters inside words)
    labse        LaBSE sentence embeddings (768 numbers per title)

The classifier is always linear, so a difference in quality comes from the
representation: a linear SVM for the sparse TF-IDF matrices and logistic
regression for the dense embeddings (the SVM solver needs more than five
minutes per fit on a dense 768-column matrix, logistic regression about one).
Hyper-parameters are chosen on a validation part of the training period; the
test set is used once, for the final numbers.
"""
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from src import config
from src.dataset import load_labelled, time_split, validation_split
from src.db import get_db_path
from src.preprocess import normalize_text, tokenize_stems, tokenize_words, vocabulary_stats
from src.utils import Timer, now_str, print_section, save_json

# Candidates with a large C can stop at max_iter before full convergence. They are
# still judged by their validation score, so the warning is only noise in the console.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
# Candidates tried on the validation part. 'balanced' gives rare classes more weight.
SPARSE_GRID = [{"C": c, "class_weight": weight}
               for c in (0.3, 1.0, 3.0) for weight in (None, "balanced")]
DENSE_GRID = [{"C": 10.0}, {"C": 100.0}]
SPARSE_MODELS = ["word_tfidf", "stem_tfidf", "char_tfidf"]
MODEL_TITLES = {
    "baseline_most_frequent": "Baseline: always the largest class",
    "baseline_stratified": "Baseline: random by class frequency",
    "word_tfidf": "Word forms TF-IDF + LinearSVC",
    "stem_tfidf": "Stems TF-IDF + LinearSVC",
    "char_tfidf": "Char n-grams TF-IDF + LinearSVC",
    "labse": "LaBSE embeddings + LogisticRegression",
}
MODELS_DIR = config.BASE_DIR / "models"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def make_vectorizer(name: str) -> TfidfVectorizer:
    """TF-IDF vectoriser for one of the sparse representations."""
    if name == "word_tfidf":
        return TfidfVectorizer(tokenizer=tokenize_words, token_pattern=None, lowercase=False,
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    elif name == "stem_tfidf":
        return TfidfVectorizer(tokenizer=tokenize_stems, token_pattern=None, lowercase=False,
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    elif name == "char_tfidf":
        return TfidfVectorizer(analyzer="char_wb", preprocessor=normalize_text, lowercase=False,
                               ngram_range=(2, 4), min_df=3, max_features=200_000,
                               sublinear_tf=True, dtype=np.float32)
    raise ValueError(f"unknown representation: {name}")


def make_classifier(params: dict, dense: bool = False):
    """Linear classifier: SVM for sparse TF-IDF features, logistic regression for embeddings."""
    if dense:
        return LogisticRegression(max_iter=1000, **params)
    # dual=True: the dual solver is 5-10 times faster than the primal one on sparse text
    return LinearSVC(dual=True, max_iter=3000, random_state=RANDOM_STATE, **params)


def describe(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def evaluate(y_true, y_pred) -> dict:
    """Quality metrics. F1-macro is averaged over the classes present in y_true."""
    labels = sorted(set(y_true))
    return {
        "f1_macro": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def matrix_megabytes(matrix) -> float:
    """Memory of a feature matrix (sparse or dense) in MB."""
    if hasattr(matrix, "indptr"):       # scipy sparse matrix
        n_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    else:
        n_bytes = matrix.nbytes
    return round(n_bytes / 1024 ** 2, 1)


def choose_params(x_fit, y_fit, x_val, y_val, dense: bool) -> tuple:
    """Try every candidate on the validation part.

    Returns (best parameters, {description: validation F1-macro}).
    """
    grid = DENSE_GRID if dense else SPARSE_GRID
    scores = {}
    for params in grid:
        classifier = make_classifier(params, dense).fit(x_fit, y_fit)
        scores[describe(params)] = evaluate(y_val, classifier.predict(x_val))["f1_macro"]
    best = max(grid, key=lambda params: scores[describe(params)])
    return best, scores


def can_validate(fit_part: pd.DataFrame, validation: pd.DataFrame) -> bool:
    """The tiny sample may have no usable validation part."""
    return len(validation) >= 20 and fit_part["section"].nunique() >= 2


# --------------------------------------------------------------------------
# One model = features + classifier
# --------------------------------------------------------------------------
def train_and_test(name: str, features, splits: dict, dense: bool = False) -> tuple:
    """Shared procedure: choose parameters on validation, refit on the whole train, test once.

    `features(fit_texts, other_texts)` returns the two matrices and, for TF-IDF,
    the fitted vectoriser. Returns (result row, test predictions, fitted model parts).
    """
    train, test = splits["train"], splits["test"]
    fit_part, validation = splits["fit_part"], splits["validation"]

    best_params, val_scores = (DENSE_GRID if dense else SPARSE_GRID)[0], {}
    if can_validate(fit_part, validation):
        x_fit, x_val, _ = features(fit_part["title"], validation["title"])
        best_params, val_scores = choose_params(x_fit, fit_part["section"],
                                                x_val, validation["section"], dense)
        del x_fit, x_val

    with Timer() as feature_timer:
        x_train, x_test, vectorizer = features(train["title"], test["title"])
    with Timer() as fit_timer:
        classifier = make_classifier(best_params, dense).fit(x_train, train["section"])
    with Timer() as predict_timer:
        predictions = classifier.predict(x_test)

    row = {
        "model": name,
        "description": MODEL_TITLES[name],
        **evaluate(test["section"], predictions),
        "val_f1_macro": val_scores.get(describe(best_params)),
        "best_params": describe(best_params),
        "features": x_train.shape[1],
        "feature_seconds": round(feature_timer.seconds, 1),
        "train_seconds": round(fit_timer.seconds, 1),
        "predict_ms_per_1000": round(1000 * predict_timer.seconds / max(len(test), 1) * 1000, 1),
        "train_matrix_mb": matrix_megabytes(x_train),
        "model_mb": round(classifier.coef_.nbytes / 1024 ** 2, 1),
        "val_scores": val_scores,
    }
    return row, predictions, {"vectorizer": vectorizer, "classifier": classifier}


def run_sparse_model(name: str, splits: dict) -> tuple:
    def features(fit_texts, other_texts):
        vectorizer = make_vectorizer(name)
        return vectorizer.fit_transform(fit_texts), vectorizer.transform(other_texts), vectorizer
    return train_and_test(name, features, splits)


def run_labse_model(splits: dict, data_name: str) -> tuple:
    from src.embeddings import encode_titles, encoding_seconds   # imported late: heavy dependency

    all_titles = pd.concat([splits["train"]["title"], splits["test"]["title"]])
    encode_titles(sorted(set(all_titles)), data_name)             # fills the cache if needed

    def features(fit_texts, other_texts):
        return encode_titles(list(fit_texts), data_name), encode_titles(list(other_texts), data_name), None

    row, predictions, parts = train_and_test("labse", features, splits, dense=True)
    # the honest feature time is the encoding itself, even when it comes from the cache
    row["feature_seconds"] = round(row["feature_seconds"] + encoding_seconds(data_name), 1)
    return row, predictions, parts


def run_baselines(splits: dict) -> list:
    train, test = splits["train"], splits["test"]
    rows = []
    for name, strategy in [("baseline_most_frequent", "most_frequent"), ("baseline_stratified", "stratified")]:
        with Timer() as fit_timer:
            dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
            dummy.fit(train[["title"]], train["section"])
        predictions = dummy.predict(test[["title"]])
        rows.append({"model": name, "description": MODEL_TITLES[name],
                     **evaluate(test["section"], predictions),
                     "features": 0, "feature_seconds": 0.0,
                     "train_seconds": round(fit_timer.seconds, 1), "train_matrix_mb": 0.0, "model_mb": 0.0})
    return rows


# --------------------------------------------------------------------------
# Saving / loading the model used for user input
# --------------------------------------------------------------------------
def model_path(data_name: str) -> Path:
    return MODELS_DIR / f"title_classifier_{data_name}.joblib"


def save_model(parts: dict, name: str, data_name: str, metrics: dict) -> Path:
    path = model_path(data_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": name, "vectorizer": parts["vectorizer"], "classifier": parts["classifier"],
                 "metrics": metrics, "trained_at": now_str()}, path, compress=3)
    return path


def load_model(data_name: str) -> dict:
    path = model_path(data_name)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Train the models first.")
    return joblib.load(path)


def predict_top_k(model: dict, title: str, k: int = 3) -> list:
    """Return the k most likely sections for a title as (section, score) pairs.

    The score is the distance to the SVM decision boundary (higher = more confident);
    it is not a probability.
    """
    if not re.search(r"[а-яәғқңөұүһі]", title.lower()):
        raise ValueError("the title has no Cyrillic letters - type a title in Kazakh")
    features = model["vectorizer"].transform([title])
    if features.nnz == 0:
        raise ValueError("the title has no known words - type a title in Kazakh")
    scores = model["classifier"].decision_function(features)[0]
    best = np.argsort(scores)[::-1][:k]
    return [(model["classifier"].classes_[i], round(float(scores[i]), 3)) for i in best]


# --------------------------------------------------------------------------
# Full comparison
# --------------------------------------------------------------------------
def prepare_splits(paths: dict) -> tuple:
    df = load_labelled(get_db_path(paths))
    train, test, report = time_split(df)
    if len(test) == 0 or train["section"].nunique() < 2:
        raise ValueError("not enough data for a train/test split")
    fit_part, validation, border = validation_split(train)
    report["validation_from"] = border
    report["validation_rows"] = len(validation)
    return {"train": train, "test": test, "fit_part": fit_part, "validation": validation}, report


def run_comparison(paths: dict, with_labse: bool = True) -> pd.DataFrame:
    """Train and test all models; save the comparison, predictions and the best fast model."""
    out_dir = paths["reports"] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_section("Train/test split")
    splits, split_report = prepare_splits(paths)
    for key, value in split_report.items():
        print(f"  {key:42s} {value}")
    save_json(split_report, out_dir / "split_report.json")

    print_section("Word forms vs stems")
    vocabulary = pd.DataFrame(vocabulary_stats(splits["train"]["title"], splits["test"]["title"]))
    print(vocabulary.to_string(index=False))
    vocabulary.to_csv(out_dir / "vocabulary_stats.csv", index=False)

    print_section("Training")
    rows = run_baselines(splits)
    predictions = splits["test"][["act_id", "title", "date_adopted", "section"]].copy()
    fitted = {}
    for name in SPARSE_MODELS:
        print(f"  {MODEL_TITLES[name]} ...")
        row, predicted, fitted[name] = run_sparse_model(name, splits)
        rows.append(row)
        predictions[f"pred_{name}"] = predicted
        print(f"    F1-macro {row['f1_macro']:.4f} | {row['best_params']} | "
              f"{row['features']:,} features | fit {row['train_seconds']} s")
    if with_labse:
        print(f"  {MODEL_TITLES['labse']} ...")
        try:
            row, predicted, _ = run_labse_model(splits, paths["name"])
            rows.append(row)
            predictions["pred_labse"] = predicted
            print(f"    F1-macro {row['f1_macro']:.4f} | {row['best_params']} | fit {row['train_seconds']} s")
        except ImportError as error:
            print(f"    skipped: {error}")

    results = pd.DataFrame(rows)
    table_columns = ["model", "f1_macro", "f1_weighted", "accuracy", "val_f1_macro", "best_params", "features",
                     "feature_seconds", "train_seconds", "predict_ms_per_1000", "train_matrix_mb", "model_mb"]
    print_section("Comparison on the test set")
    print(results[table_columns].to_string(index=False))
    results.drop(columns=["val_scores"]).to_csv(out_dir / "model_comparison.csv", index=False)
    save_json(rows, out_dir / "model_comparison.json")
    predictions.to_parquet(out_dir / "test_predictions.parquet", index=False)

    # The model for user input must answer instantly on any laptop, so it is chosen
    # among the TF-IDF models - by validation F1 when there is one, otherwise by test F1.
    sparse = results[results["model"].isin(SPARSE_MODELS)]
    criterion = "val_f1_macro" if sparse["val_f1_macro"].notna().any() else "f1_macro"
    chosen = sparse.sort_values(criterion, ascending=False).iloc[0]
    saved = save_model(fitted[chosen["model"]], chosen["model"], paths["name"],
                       {"f1_macro": chosen["f1_macro"], "accuracy": chosen["accuracy"]})
    print(f"\nModel for user input: {chosen['description']} (chosen by {criterion}), saved to {saved}")
    print(f"Results saved to {out_dir}")
    return results
