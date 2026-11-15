"""Kazakh Legal Act Classification - console application.

Interactive menu:      python main.py
Small built-in sample: python main.py --sample
Without the menu:      python main.py --sample --action all
"""
import argparse

from src import audit, config, data_loader
from src.utils import format_bytes, load_json, print_section, save_json

MENU = {
    "1": "Download the dataset from Hugging Face (~337 MB)",
    "2": "Convert CSV -> Parquet in chunks (cleaning + memory report)",
    "3": "Memory by column: before / after dtype optimisation",
    "4": "Benchmark: CSV vs Parquet (time, memory, file size)",
    "5": "Metadata audit (missing, placeholders, dates, multi-valued, duplicates)",
    "6": "Dataset overview",
    "7": "Switch between FULL data and the 500-row SAMPLE",
    "0": "Exit",
}


# --------------------------------------------------------------------------
# Actions (one function per menu item)
# --------------------------------------------------------------------------
def action_download(paths: dict) -> None:
    if paths["name"] == "sample":
        print("The sample is already in the repository - nothing to download.")
        return
    data_loader.download_dataset()


def action_convert(paths: dict) -> None:
    print_section(f"CSV -> Parquet ({paths['name']} data)")
    report = data_loader.convert_csv_to_parquet(paths["csv"], paths["parquet"])
    data_loader.print_conversion_report(report)
    saved = save_json(report, paths["reports"] / "conversion_report.json")
    print(f"Report saved to {saved}")


def action_memory(paths: dict) -> None:
    print_section("Memory by column: before / after dtype optimisation")
    table = data_loader.dtype_memory_table(paths["parquet"])
    print(table.to_string())
    before, after = table["mb_before"].sum(), table["mb_after"].sum()
    print(f"\nTOTAL: {before:.1f} MB -> {after:.1f} MB ({100 * (1 - after / before):.1f}% saved)")
    paths["reports"].mkdir(parents=True, exist_ok=True)
    table.rename_axis("column").reset_index().to_csv(paths["reports"] / "dtype_memory.csv", index=False)


def action_benchmark(paths: dict) -> None:
    print_section("Benchmark: CSV vs Parquet")
    if not paths["csv"].exists():
        raise FileNotFoundError(f"CSV file not found: {paths['csv']}")
    table = data_loader.benchmark_formats(paths["csv"], paths["parquet"])
    print(table.to_string(index=False))
    paths["reports"].mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["reports"] / "benchmark_formats.csv", index=False)


def action_audit(paths: dict) -> None:
    df = data_loader.load_acts(paths["parquet"])
    conversion_report = load_json(paths["reports"] / "conversion_report.json")
    audit.run_full_audit(df, conversion_report, paths["reports"] / "audit")


def action_overview(paths: dict) -> None:
    print_section(f"Dataset overview ({paths['name']} data)")
    df = data_loader.load_acts(paths["parquet"])
    print(f"Rows: {len(df):,} | columns: {df.shape[1]} | "
          f"memory: {format_bytes(data_loader.memory_bytes(df))}")
    if "date_adopted" in df.columns:
        print(f"Adopted between {df['date_adopted'].min().date()} and {df['date_adopted'].max().date()}")
    for col in ["database_section", "act_form"]:
        if col in df.columns:
            print(f"\nTop 5 values of '{col}' ({df[col].nunique()} distinct):")
            for value, count in df[col].value_counts().head(5).items():
                print(f"  {count:>7,}  {value}")
    print("\nThree random titles:")
    for title in df[config.TEXT_COLUMN].dropna().sample(min(3, len(df)), random_state=42):
        print(f"  - {title[:110]}")


ACTIONS = {
    "1": action_download,
    "2": action_convert,
    "3": action_memory,
    "4": action_benchmark,
    "5": action_audit,
    "6": action_overview,
}
# names for the --action command-line option
ACTION_NAMES = {"download": "1", "convert": "2", "memory": "3",
                "benchmark": "4", "audit": "5", "overview": "6"}
PIPELINE = ["convert", "memory", "benchmark", "audit", "overview"]   # --action all


def run_action(key: str, paths: dict) -> bool:
    """Run one action; errors are reported instead of crashing the program."""
    try:
        ACTIONS[key](paths)
        return True
    except FileNotFoundError as error:
        print(f"\nFile is missing: {error}")
        if paths["name"] == "full":
            print("Hint: run item 1 (download) and item 2 (convert) first, "
                  "or switch to the sample with item 7.")
        else:
            print("Hint: run item 2 (convert) first.")
    except (ValueError, KeyError) as error:
        print(f"\nThe data has an unexpected format: {error}")
    except OSError as error:
        print(f"\nDownload or disk problem: {error}")
    return False


def show_menu(paths: dict) -> None:
    print()
    print("-" * 78)
    print(f"KAZAKH LEGAL ACT CLASSIFICATION   [data: {paths['name'].upper()}]")
    print("-" * 78)
    for key, label in MENU.items():
        print(f"  {key} - {label}")


def menu_loop(use_sample: bool) -> None:
    """Interactive console menu; repeats until the user chooses 0."""
    paths = config.get_paths(use_sample)
    while True:
        show_menu(paths)
        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if choice == "0":
            print("Bye!")
            break
        elif choice == "7":
            use_sample = not use_sample
            paths = config.get_paths(use_sample)
            print(f"Now working with the {paths['name'].upper()} data: {paths['csv']}")
        elif choice in ACTIONS:
            run_action(choice, paths)
        else:
            print(f"Unknown option '{choice}'. Enter one of: {', '.join(MENU)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kazakh Legal Act Classification")
    parser.add_argument("--sample", action="store_true",
                        help="work with the 500-row sample instead of the full dataset")
    parser.add_argument("--action", choices=list(ACTION_NAMES) + ["all"],
                        help="run one step without the menu ('all' = full pipeline)")
    args = parser.parse_args()

    if args.action is None:
        menu_loop(args.sample)
        return

    paths = config.get_paths(args.sample)
    names = PIPELINE if args.action == "all" else [args.action]
    for name in names:
        if not run_action(ACTION_NAMES[name], paths):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
