"""Exploratory data analysis: charts (PNG + one interactive HTML) and findings.

Aggregated numbers come from the SQLite database (SQL queries), title texts
from the Parquet file. Every plot function saves its chart and returns one
sentence with the numbers behind it; the sentences are collected in
eda_findings.md, so each conclusion is tied to the data.
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data_loader import load_acts
from src.db import get_db_path, run_query
from src.queries import SPLIT_DATE, find_query, run_one
from src.utils import print_section, save_json

# Colours: one hue for single-series charts, a second one only for "test".
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

WORD_PATTERN = re.compile(r"[а-яёәғқңөұүһі]+")     # Russian + Kazakh Cyrillic letters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def setup_style() -> None:
    """Common look of all charts."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": TEXT, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


def shorten(text: str, max_length: int = 48) -> str:
    """Cut long Kazakh section names so that axis labels stay readable."""
    text = str(text)
    return text if len(text) <= max_length else text[:max_length - 1].rstrip() + "…"


def set_title(fig, text: str) -> None:
    """Title aligned to the left edge of the figure (long axis labels cannot clip it)."""
    fig.suptitle(text, x=0.012, ha="left", fontsize=12, fontweight="bold", color=TEXT)


def save_figure(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def horizontal_bars(ax, labels, values, color=BLUE, value_format="{:,.0f}") -> None:
    """Horizontal bar chart, largest value on top, numbers at the bar ends."""
    positions = range(len(labels))
    ax.barh(positions, values, color=color, height=0.72)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for position, value in zip(positions, values):
        ax.text(value, position, " " + value_format.format(value), va="center", fontsize=8, color=MUTED)
    ax.margins(x=0.12)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_class_distribution(db_path: Path, out_dir: Path) -> str:
    """1. How many acts each section (target class) has."""
    data, _ = run_one(db_path, find_query("q01_sections"))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.27 * len(data) + 1)))
    horizontal_bars(ax, [shorten(s) for s in data["section"]], data["acts"])
    set_title(fig, "Target classes are strongly imbalanced")
    ax.set_xlabel("Number of acts")
    save_figure(fig, out_dir, "01_class_distribution")

    top, last = data.iloc[0], data.iloc[-1]
    top3_share = data["share_pct"].head(3).sum()
    return (f"Class imbalance: {len(data)} sections; the largest ('{top['section']}') has "
            f"{top['acts']:,} acts ({top['share_pct']}%), the smallest ('{last['section']}') {last['acts']:,}. "
            f"The top three sections cover {top3_share:.1f}% of all acts, so accuracy would be "
            f"misleading and F1-macro is used.")


def plot_yearly_dynamics(db_path: Path, out_dir: Path) -> str:
    """2. Acts per year with a 3-year moving average and the split date."""
    data, _ = run_one(db_path, find_query("q03_yearly_dynamics"))
    split_year = int(SPLIT_DATE[:4])
    last_full_year = int(data["year"].max()) - 1          # the last year in the corpus is incomplete
    data = data[data["year"] <= last_full_year]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(data["year"], data["acts"], color=BLUE, alpha=0.35, width=0.8, label="Acts per year")
    ax.plot(data["year"], data["moving_avg_3y"], color=BLUE, linewidth=2, label="3-year moving average")
    ax.set_ylim(0, data["acts"].max() * 1.15)             # head-room for the label
    ax.axvline(split_year - 0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.text(split_year - 0.7, ax.get_ylim()[1] * 0.98, f"test period: from {split_year} ",
            color=MUTED, fontsize=9, va="top", ha="right")
    set_title(fig, "Number of adopted acts per year")
    ax.set_xlabel(f"Year of adoption (the incomplete year {last_full_year + 1} is not shown)")
    ax.set_ylabel("Number of acts")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, out_dir, "02_acts_per_year")

    peak = data.loc[data["acts"].idxmax()]
    test_share = 100 * data.loc[data["year"] >= split_year, "acts"].sum() / data["acts"].sum()
    return (f"Time: the yearly number of acts grows from {int(data['acts'].iloc[0]):,} in "
            f"{int(data['year'].iloc[0])} to a peak of {int(peak['acts']):,} in {int(peak['year'])}. "
            f"Acts adopted from {split_year} make up {test_share:.1f}% of the shown period, "
            f"which gives a test set of a reasonable size for a time-based split.")


def plot_title_length(df: pd.DataFrame, out_dir: Path) -> tuple:
    """3. Length of titles in tokens + outliers by the IQR rule."""
    tokens = df[config.TEXT_COLUMN].str.split().str.len()
    q1, median, q3 = tokens.quantile([0.25, 0.5, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    outliers = int((tokens > upper_fence).sum())
    shown = tokens[tokens <= tokens.quantile(0.99)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(shown, bins=range(1, int(shown.max()) + 2), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(median, color=TEXT, linewidth=1.5)
    ax.text(median + 0.6, ax.get_ylim()[1] * 0.95, f"median = {median:.0f} words", va="top", fontsize=9)
    ax.axvline(upper_fence, color=MUTED, linestyle="--", linewidth=1)
    ax.text(upper_fence + 0.6, ax.get_ylim()[1] * 0.80, f"IQR outlier limit = {upper_fence:.0f}",
            va="top", fontsize=9, color=MUTED)
    set_title(fig, "Titles are short texts: length in words")
    ax.set_xlabel("Words in the title (the longest 1% of titles are not shown)")
    ax.set_ylabel("Number of acts")
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir, "03_title_length")

    stats = {
        "tokens_min": int(tokens.min()), "tokens_q1": float(q1), "tokens_median": float(median),
        "tokens_mean": round(float(tokens.mean()), 1), "tokens_q3": float(q3),
        "tokens_max": int(tokens.max()), "iqr_upper_fence": float(upper_fence),
        "iqr_outliers": outliers, "one_word_titles": int((tokens == 1).sum()),
    }
    finding = (f"Title length: median {median:.0f} words (IQR {q1:.0f}-{q3:.0f}), maximum {tokens.max():,}. "
               f"{outliers:,} titles ({100 * outliers / len(tokens):.1f}%) are longer than the IQR limit of "
               f"{upper_fence:.0f} words and {stats['one_word_titles']:,} consist of a single word. "
               f"Texts are short, so sparse TF-IDF features are a sensible first choice.")
    return finding, stats


def plot_top_words(df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> str:
    """4. Most frequent words in titles (before any stop-word removal)."""
    counts = Counter()
    for title in df[config.TEXT_COLUMN].dropna():
        counts.update(WORD_PATTERN.findall(title.lower()))
    top = counts.most_common(top_n)
    words, frequencies = [w for w, _ in top], [n for _, n in top]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    horizontal_bars(ax, words, frequencies)
    set_title(fig, f"The {top_n} most frequent words in titles")
    ax.set_xlabel("Occurrences in all titles")
    save_figure(fig, out_dir, "04_top_words")

    total = sum(counts.values())
    top_share = 100 * sum(frequencies) / total
    return (f"Vocabulary: {len(counts):,} distinct word forms in {total:,} words. The {top_n} most frequent "
            f"words ('{words[0]}', '{words[1]}', '{words[2]}', ...) give {top_share:.1f}% of all words and "
            f"are mostly function and formula words, so stop-word removal and IDF weighting are needed. "
            f"{sum(1 for n in counts.values() if n == 1):,} word forms occur only once - a sign of rich "
            f"Kazakh morphology that stemming or character n-grams should reduce.")


def plot_section_period_heatmap(db_path: Path, out_dir: Path, top_n: int = 12) -> str:
    """5. Share of each section inside every five-year period (class drift)."""
    data = run_query(db_path, """
        SELECT (CAST(strftime('%Y', a.date_adopted) AS INTEGER) / 5) * 5 AS period_start,
               s.name AS section, COUNT(*) AS acts
        FROM acts a
        JOIN database_sections s ON s.section_id = a.section_id
        WHERE a.date_adopted >= '1995-01-01'
        GROUP BY period_start, s.name;
    """)
    table = data.pivot_table(index="section", columns="period_start", values="acts", fill_value=0)
    shares = 100 * table / table.sum(axis=0)
    top_sections = table.sum(axis=1).sort_values(ascending=False).head(top_n).index
    shares = shares.loc[top_sections]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(shares) + 1.6))
    image = ax.imshow(shares.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(shares.shape[1]), [f"{p}–{str(p + 4)[2:]}" for p in shares.columns], fontsize=8.5)
    ax.set_yticks(range(len(shares)), [shorten(s, 38) for s in shares.index])
    ax.grid(visible=False)
    for row in range(shares.shape[0]):
        for col in range(shares.shape[1]):
            value = shares.values[row, col]
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if value > shares.values.max() * 0.55 else TEXT)
    set_title(fig, "The mix of sections changes over time")
    ax.set_xlabel("Period of adoption (the last period is incomplete)")
    fig.colorbar(image, ax=ax, label="Share of the period's acts, %", pad=0.02)
    save_figure(fig, out_dir, "05_section_by_period")

    leader = shares.index[0]
    first, last = shares.columns[0], shares.columns[-1]
    return (f"Class drift: the share of '{leader}' changes from {shares.loc[leader, first]:.0f}% in "
            f"{first}-{first + 4} to {shares.loc[leader, last]:.0f}% in the latest period. The class mix is "
            f"not stable in time, so a random split would be too optimistic; the split is done by date.")


def plot_train_test_shift(db_path: Path, out_dir: Path, top_n: int = 12) -> str:
    """6. Class shares in the training period vs the test period."""
    data, _ = run_one(db_path, find_query("q11_split_by_section"))
    data["train_pct"] = 100 * data["train_acts"] / data["train_acts"].sum()
    data["test_pct"] = 100 * data["test_acts"] / max(data["test_acts"].sum(), 1)
    shown = data.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(shown) + 1.5))
    positions = range(len(shown))
    ax.barh([p - 0.2 for p in positions], shown["train_pct"], height=0.36, color=BLUE,
            label=f"Train (before {SPLIT_DATE[:4]})")
    ax.barh([p + 0.2 for p in positions], shown["test_pct"], height=0.36, color=ORANGE,
            label=f"Test (from {SPLIT_DATE[:4]})")
    ax.set_yticks(positions, [shorten(s, 44) for s in shown["section"]])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    set_title(fig, "Class shares differ between the train and the test period")
    ax.set_xlabel("Share of acts in the period, %")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, out_dir, "06_train_test_shift")

    data["shift"] = data["test_pct"] - data["train_pct"]
    grown, shrunk = data.loc[data["shift"].idxmax()], data.loc[data["shift"].idxmin()]
    rare = int((data["test_acts"] < 20).sum())
    return (f"Train/test shift: '{grown['section']}' grows from {grown['train_pct']:.1f}% to "
            f"{grown['test_pct']:.1f}% of acts, '{shrunk['section']}' falls from {shrunk['train_pct']:.1f}% "
            f"to {shrunk['test_pct']:.1f}%. {rare} sections have fewer than 20 test acts, so their "
            f"per-class F1 will be unstable.")


def plot_top_areas(db_path: Path, out_dir: Path, top_n: int = 15) -> str:
    """7. Most frequent areas of legal regulation (multi-valued field)."""
    data, _ = run_one(db_path, find_query("q06_areas"))
    data = data.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    horizontal_bars(ax, [shorten(a, 52) for a in data["area"]], data["pct_of_all_acts"], value_format="{:.1f}%")
    set_title(fig, "One technical area is attached to half of the acts")
    ax.set_xlabel("Share of all acts that have the area, %")
    save_figure(fig, out_dir, "07_top_areas")

    per_act = run_query(db_path, """
        SELECT ROUND(AVG(n), 2) AS mean_areas, MAX(n) AS max_areas,
               ROUND(100.0 * AVG(n > 1), 1) AS multi_pct
        FROM (SELECT COUNT(*) AS n FROM act_areas GROUP BY act_id);
    """).iloc[0]
    return (f"Areas of regulation: an act has {per_act['mean_areas']} areas on average (up to "
            f"{int(per_act['max_areas'])}); {per_act['multi_pct']}% of acts have more than one. The most "
            f"frequent area covers {data['pct_of_all_acts'].iloc[0]}% of acts and describes the procedure, "
            f"not the topic. That is why the single-valued 'database_section' is the target and the "
            f"multi-valued field would need a multi-label model.")


def plot_interactive_sections(db_path: Path, out_dir: Path, top_n: int = 6):
    """8. Interactive Plotly chart: acts per year for the largest sections."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly is not installed - the interactive chart is skipped")
        return None

    data = run_query(db_path, """
        SELECT CAST(strftime('%Y', a.date_adopted) AS INTEGER) AS year, s.name AS section, COUNT(*) AS acts
        FROM acts a
        JOIN database_sections s ON s.section_id = a.section_id
        WHERE a.date_adopted >= '1995-01-01'
        GROUP BY year, s.name;
    """)
    data = data[data["year"] < data["year"].max()]            # drop the incomplete last year
    top_sections = data.groupby("section")["acts"].sum().sort_values(ascending=False).head(top_n).index

    fig = go.Figure()
    for color, section in zip(SERIES_COLORS, top_sections):
        part = data[data["section"] == section].sort_values("year")
        fig.add_trace(go.Scatter(x=part["year"], y=part["acts"], mode="lines+markers",
                                 name=shorten(section, 40), line={"color": color, "width": 2},
                                 marker={"size": 5}))
    fig.update_layout(
        title="Acts per year in the six largest sections (click a legend item to hide it)",
        xaxis_title="Year of adoption", yaxis_title="Number of acts",
        hovermode="x unified", template="plotly_white",
        legend={"orientation": "h", "y": -0.2}, height=560,
    )
    path = Path(out_dir) / "08_sections_per_year_interactive.html"
    fig.write_html(path, include_plotlyjs=True)
    return path


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------
def run_eda(paths: dict) -> list:
    """Create all charts, print the findings and save them to eda_findings.md."""
    out_dir = paths["reports"] / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = get_db_path(paths)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Build the database first.")
    df = load_acts(paths["parquet"], columns=[config.TEXT_COLUMN], optimize=False)
    setup_style()

    findings = [
        plot_class_distribution(db_path, out_dir),
        plot_yearly_dynamics(db_path, out_dir),
    ]
    length_finding, length_stats = plot_title_length(df, out_dir)
    findings.append(length_finding)
    findings.append(plot_top_words(df, out_dir))
    findings.append(plot_section_period_heatmap(db_path, out_dir))
    findings.append(plot_train_test_shift(db_path, out_dir))
    findings.append(plot_top_areas(db_path, out_dir))
    interactive_path = plot_interactive_sections(db_path, out_dir)

    print_section("EDA findings")
    for number, finding in enumerate(findings, start=1):
        print(f"{number}. {finding}\n")

    lines = [f"# EDA findings ({paths['name']} data)", ""]
    lines += [f"{number}. {finding}" for number, finding in enumerate(findings, start=1)]
    (out_dir / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(length_stats, out_dir / "title_length_stats.json")

    print(f"Charts saved to {out_dir}")
    if interactive_path is not None:
        print(f"Interactive chart: open {interactive_path} in a browser")
    return findings
