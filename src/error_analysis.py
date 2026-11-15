"""Error analysis of the trained models on the test set.

Reads the files written by models.run_comparison (test predictions and the
comparison table) and answers: which classes are hard, which classes are
confused with each other, which titles are misclassified and why.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.eda import BLUE, MUTED, TEXT, horizontal_bars, save_figure, set_title, setup_style, shorten
from src.models import MODEL_TITLES
from src.utils import print_section, save_json

LENGTH_BINS = [0, 5, 10, 20, 30, 10_000]
LENGTH_LABELS = ["1-5", "6-10", "11-20", "21-30", "31+"]


def load_results(models_dir: Path) -> tuple:
    """Return (predictions DataFrame, comparison DataFrame)."""
    predictions_path = Path(models_dir) / "test_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions in {models_dir}. Train the models first.")
    predictions = pd.read_parquet(predictions_path)
    comparison = pd.read_csv(Path(models_dir) / "model_comparison.csv")
    return predictions, comparison


def best_model_name(comparison: pd.DataFrame, predictions: pd.DataFrame) -> str:
    """The model with the highest test F1-macro among those that have predictions."""
    available = comparison[comparison["model"].map(lambda m: f"pred_{m}" in predictions.columns)]
    return available.sort_values("f1_macro", ascending=False).iloc[0]["model"]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and number of test acts for every class."""
    labels = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.loc[labels].rename(columns={"f1-score": "f1"})
    table["support"] = table["support"].astype(int)
    return table.round(3).rename_axis("section").reset_index().sort_values("f1")


def confused_pairs(y_true, y_pred, top_n: int = 15) -> pd.DataFrame:
    """Most frequent (true class -> predicted class) mistakes."""
    errors = pd.DataFrame({"true": y_true, "predicted": y_pred})
    errors = errors[errors["true"] != errors["predicted"]]
    pairs = errors.value_counts().rename("errors").reset_index()
    class_sizes = pd.Series(y_true).value_counts()
    pairs["pct_of_true_class"] = (100 * pairs["errors"] / pairs["true"].map(class_sizes)).round(1)
    return pairs.head(top_n)


def errors_by_length(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Error rate for short and long titles."""
    words = predictions["title"].str.split().str.len()
    bucket = pd.cut(words, bins=LENGTH_BINS, labels=LENGTH_LABELS)
    wrong = predictions["section"] != predictions[column]
    table = wrong.groupby(bucket, observed=True).agg(acts="size", errors="sum")
    table["error_rate_pct"] = (100 * table["errors"] / table["acts"]).round(1)
    return table.rename_axis("words_in_title").reset_index()


def model_agreement(predictions: pd.DataFrame, best_column: str) -> dict:
    """How many errors of the best model are shared with all the other models."""
    columns = [c for c in predictions.columns if c.startswith("pred_")]
    wrong = pd.DataFrame({c: predictions[c] != predictions["section"] for c in columns})
    best_wrong = wrong[best_column]
    others = [c for c in columns if c != best_column]
    return {
        "test_acts": len(predictions),
        "errors_of_best_model": int(best_wrong.sum()),
        "errors_shared_by_all_models": int(wrong.all(axis=1).sum()),
        "errors_another_model_gets_right": int((best_wrong & ~wrong[others].all(axis=1)).sum()) if others else 0,
        "acts_all_models_get_right": int((~wrong).all(axis=1).sum()),
    }


def bootstrap_f1(predictions: pd.DataFrame, n_rounds: int = 200, seed: int = 42) -> pd.DataFrame:
    """95% interval of F1-macro for every model.

    The test set is resampled with replacement `n_rounds` times. If the intervals
    of two models overlap, the difference between them is not reliable.
    """
    columns = [c for c in predictions.columns if c.startswith("pred_")]
    labels, y_true = np.unique(predictions["section"], return_inverse=True)
    code = {label: i for i, label in enumerate(labels)}
    encoded = {c: predictions[c].map(code).fillna(-1).astype(int).to_numpy() for c in columns}
    generator = np.random.default_rng(seed)
    scores = {c: [] for c in columns}
    for _ in range(n_rounds):
        rows = generator.integers(0, len(y_true), len(y_true))
        present = np.unique(y_true[rows])
        for c in columns:
            scores[c].append(f1_score(y_true[rows], encoded[c][rows], labels=present,
                                      average="macro", zero_division=0))
    table = pd.DataFrame([{
        "model": c.replace("pred_", ""),
        "f1_macro_low": round(float(np.percentile(scores[c], 2.5)), 4),
        "f1_macro_mean": round(float(np.mean(scores[c])), 4),
        "f1_macro_high": round(float(np.percentile(scores[c], 97.5)), 4),
    } for c in columns])
    return table.sort_values("f1_macro_mean", ascending=False)


def error_examples(predictions: pd.DataFrame, column: str, pairs: pd.DataFrame, per_pair: int = 3) -> pd.DataFrame:
    """A few misclassified titles for each of the most frequent confusions."""
    examples = []
    for _, pair in pairs.head(8).iterrows():
        mask = (predictions["section"] == pair["true"]) & (predictions[column] == pair["predicted"])
        part = predictions.loc[mask, ["title", "section", column, "date_adopted"]].head(per_pair)
        examples.append(part.rename(columns={"section": "true", column: "predicted"}))
    return pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path) -> None:
    data = comparison.sort_values("f1_macro", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(data) + 1.6))
    colors = [MUTED if name.startswith("baseline") else BLUE for name in data["model"]]
    horizontal_bars(ax, [MODEL_TITLES.get(m, m) for m in data["model"]], data["f1_macro"],
                    color=colors, value_format="{:.3f}")
    ax.set_xlim(0, 1)
    set_title(fig, "F1-macro on the test set (grey = baselines)")
    ax.set_xlabel("F1-macro (1.0 = perfect)")
    save_figure(fig, out_dir, "model_comparison")


def plot_per_class_f1(table: pd.DataFrame, model_title: str, out_dir: Path) -> None:
    data = table.sort_values("f1", ascending=False)
    labels = [f"{shorten(s, 40)}  (n={n:,})" for s, n in zip(data["section"], data["support"])]
    fig, ax = plt.subplots(figsize=(9, 0.27 * len(data) + 1.6))
    horizontal_bars(ax, labels, data["f1"], value_format="{:.2f}")
    ax.set_xlim(0, 1.08)
    set_title(fig, f"F1 by section, n = test acts ({model_title})")
    ax.set_xlabel("F1 of the class")
    save_figure(fig, out_dir, "f1_by_class")


def plot_confusion_matrix(y_true, y_pred, model_title: str, out_dir: Path, top_n: int = 12) -> None:
    """Row-normalised confusion matrix of the largest classes; the rest is 'Other'."""
    top = list(pd.Series(y_true).value_counts().head(top_n).index)
    fold = lambda labels: [label if label in top else "Other sections" for label in labels]
    labels = top + ["Other sections"]
    matrix = confusion_matrix(fold(y_true), fold(y_pred), labels=labels)
    shares = 100 * matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=100)
    short = [shorten(label, 26) for label in labels]
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    ax.grid(visible=False)
    for row in range(len(labels)):
        for col in range(len(labels)):
            if shares[row, col] >= 1:
                ax.text(col, row, f"{shares[row, col]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if shares[row, col] > 55 else TEXT)
    set_title(fig, f"Confusion matrix, % of the true class ({model_title})")
    ax.set_xlabel("Predicted section")
    ax.set_ylabel("True section")
    fig.colorbar(image, ax=ax, label="% of acts of the true section", shrink=0.8)
    save_figure(fig, out_dir, "confusion_matrix")


def plot_errors_by_length(table: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(table["words_in_title"].astype(str), table["error_rate_pct"], color=BLUE, width=0.6)
    for x, (rate, acts) in enumerate(zip(table["error_rate_pct"], table["acts"])):
        ax.text(x, rate, f"{rate:.1f}%\nn={acts:,}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.margins(y=0.25)
    ax.grid(axis="x", visible=False)
    set_title(fig, "Error rate by title length")
    ax.set_xlabel("Words in the title")
    ax.set_ylabel("Misclassified acts, %")
    save_figure(fig, out_dir, "errors_by_length")


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------
def run_error_analysis(paths: dict, model: str = None) -> dict:
    models_dir = paths["reports"] / "models"
    out_dir = paths["reports"] / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions, comparison = load_results(models_dir)
    model = model or best_model_name(comparison, predictions)
    column = f"pred_{model}"
    if column not in predictions.columns:
        raise KeyError(f"no predictions for model '{model}'")
    title = MODEL_TITLES.get(model, model)
    y_true, y_pred = predictions["section"], predictions[column]
    setup_style()

    print_section(f"Error analysis: {title}")
    per_class = per_class_table(y_true, y_pred)
    pairs = confused_pairs(y_true, y_pred)
    by_length = errors_by_length(predictions, column)
    agreement = model_agreement(predictions, column)
    examples = error_examples(predictions, column, pairs)
    intervals = bootstrap_f1(predictions)

    print("F1-macro with 95% bootstrap intervals (200 resamples of the test set):")
    print(intervals.to_string(index=False))
    print()

    print("Hardest classes:")
    print(per_class.head(8).to_string(index=False, max_colwidth=45))
    print("\nEasiest classes:")
    print(per_class.tail(5).to_string(index=False, max_colwidth=45))
    print("\nMost frequent confusions (true -> predicted):")
    print(pairs.head(10).to_string(index=False, max_colwidth=42))
    print("\nError rate by title length:")
    print(by_length.to_string(index=False))
    print("\nAgreement between models:")
    for key, value in agreement.items():
        print(f"  {key:34s} {value:,}")
    print("\nExamples of misclassified titles:")
    for _, row in examples.head(8).iterrows():
        print(f"  - {shorten(row['title'], 100)}\n      true: {shorten(row['true'], 45)} | predicted: {shorten(row['predicted'], 45)}")

    intervals.to_csv(out_dir / "f1_intervals.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    pairs.to_csv(out_dir / "confused_pairs.csv", index=False)
    by_length.to_csv(out_dir / "errors_by_length.csv", index=False)
    examples.to_csv(out_dir / "error_examples.csv", index=False)
    plot_model_comparison(comparison, out_dir)
    plot_per_class_f1(per_class, title, out_dir)
    plot_confusion_matrix(list(y_true), list(y_pred), title, out_dir)
    plot_errors_by_length(by_length, out_dir)

    # correlation between class size in the test set and its F1
    rare = per_class[per_class["support"] < 50]
    summary = {
        "model": model,
        **agreement,
        "classes": len(per_class),
        "classes_with_f1_below_0.5": int((per_class["f1"] < 0.5).sum()),
        "mean_f1_classes_under_50_test_acts": round(float(rare["f1"].mean()), 3) if len(rare) else None,
        "mean_f1_classes_from_50_test_acts": round(float(per_class[per_class["support"] >= 50]["f1"].mean()), 3),
        "top_confusion": pairs.iloc[0].to_dict() if len(pairs) else None,
    }
    save_json(summary, out_dir / "error_summary.json")
    print(f"\nTables and charts saved to {out_dir}")
    return summary
