"""Sentence embeddings of titles with LaBSE (Language-agnostic BERT Sentence Embedding).

Encoding 160k titles takes several minutes, so the vectors are cached on disk:
    data/processed/labse_<name>.npy        matrix [n_titles, 768], float16
    data/processed/labse_<name>_titles.json  the titles, in the same order
    data/processed/labse_<name>_meta.json    how long the encoding took
"""
import json
from pathlib import Path

import numpy as np

from src import config
from src.utils import Timer

MODEL_NAME = "sentence-transformers/LaBSE"
MAX_TOKENS = 96          # titles are short; longer ones are cut
BATCH_SIZE = 64


def cache_paths(name: str) -> tuple:
    base = config.PROCESSED_DIR / f"labse_{name}"
    return base.with_suffix(".npy"), Path(str(base) + "_titles.json")


def meta_path(name: str) -> Path:
    return config.PROCESSED_DIR / f"labse_{name}_meta.json"


def encoding_seconds(name: str) -> float:
    """Total time that was spent on encoding the cached titles (0 if unknown)."""
    if not meta_path(name).exists():
        return 0.0
    with open(meta_path(name), "r", encoding="utf-8") as f:
        return float(json.load(f).get("encode_seconds", 0.0))


def load_cache(name: str) -> dict:
    """Return {title: row index} and the matrix, or ({}, None) if there is no cache."""
    matrix_path, titles_path = cache_paths(name)
    if not matrix_path.exists() or not titles_path.exists():
        return {}, None
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = json.load(f)
    matrix = np.load(matrix_path)
    return {title: i for i, title in enumerate(titles)}, matrix


def encode_titles(titles: list, name: str) -> np.ndarray:
    """Return one LaBSE vector per title (float32). Uses and updates the disk cache."""
    index, matrix = load_cache(name)
    missing = sorted({title for title in titles if title not in index})

    if missing:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("LaBSE needs: pip install sentence-transformers") from error

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():      # Apple Silicon GPU
            device = "mps"
        else:
            device = "cpu"
        print(f"  LaBSE: encoding {len(missing):,} new titles on '{device}' "
              f"({len(titles) - len(missing):,} taken from the cache)...")
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = MAX_TOKENS
        with Timer() as timer:
            new_vectors = model.encode(missing, batch_size=BATCH_SIZE, show_progress_bar=True,
                                       normalize_embeddings=True, convert_to_numpy=True)
        print(f"  encoded in {timer.seconds:.0f} s ({len(missing) / timer.seconds:.0f} titles per second)")

        new_vectors = new_vectors.astype(np.float16)
        all_titles = list(index) + missing
        matrix = new_vectors if matrix is None else np.vstack([matrix, new_vectors])
        index = {title: i for i, title in enumerate(all_titles)}

        matrix_path, titles_path = cache_paths(name)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, matrix)
        with open(titles_path, "w", encoding="utf-8") as f:
            json.dump(all_titles, f, ensure_ascii=False)
        # read the previous total BEFORE opening the file for writing: "w" empties it
        total_seconds = round(encoding_seconds(name) + timer.seconds, 1)
        with open(meta_path(name), "w", encoding="utf-8") as f:
            json.dump({"model": MODEL_NAME, "device": device, "titles": len(all_titles),
                       "encode_seconds": total_seconds}, f)

    rows = [index[title] for title in titles]
    return matrix[rows].astype(np.float32)
