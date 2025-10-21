from pathlib import Path
import os
from typing import Union, Tuple

import click
import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_pisa import PisaIndex

from suiteeval.context import DatasetContext
from suiteeval import BEIR


def _dir_size_bytes(path: Union[str, os.PathLike]) -> int:
    """Return total size in bytes for a file or directory (recursive)."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    stack = [p]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                except FileNotFoundError:
                    continue
    return total


def _mb(x_bytes: int) -> float:
    return x_bytes / (1024.0 ** 2)


def _parse_grid(grid_str: str) -> Tuple[float, ...]:
    """Parse a comma-separated list of floats into a tuple."""
    vals = []
    for tok in grid_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise click.BadParameter("Grid cannot be empty.")
    return tuple(vals)


@click.command()
@click.option("--save-path", type=str, default="results.bm25.csv.gz",
              help="Path to save the CSV results.")
@click.option("--k1-grid", type=str, default="0.1,0.3,0.5,1.0,1.2,1.5,2.0",
              help="Comma-separated k1 values to sweep, e.g. '0.7,0.9,1.2'.")
@click.option("--b-grid", type=str, default="0.3,0.5,0.75,0.9",
              help="Comma-separated b values to sweep, e.g. '0.3,0.5,0.75'.")
def main(
    save_path: str,
    k1_grid: str,
    b_grid: str,
):
    k1_values = _parse_grid(k1_grid)
    b_values = _parse_grid(b_grid)

    def pipelines(context: DatasetContext):
        # --- PISA (BM25) indexing ---
        pisa_dir = f"{context.path}/index.pisa"
        pisa_index = PisaIndex(pisa_dir)
        pisa_index.index(context.get_corpus_iter())

        # Compute on-disk size for PISA index (constant across sweeps)
        pisa_size_b = _dir_size_bytes(pisa_dir)
        pisa_size_mb = _mb(pisa_size_b)

        # --- Sweep BM25 hyperparameters ---
        for k1 in k1_values:
            for b in b_values:
                bm25 = pisa_index.bm25(k1=k1, b=b)
                label = (
                    f"BM25 |k1={k1:g}|b={b:g}|"
                    f" |size={pisa_size_b}| ({pisa_size_mb:.1f} MB)"
                )
                yield (bm25, label)

    # Run the evaluation suite
    result = BEIR(pipelines)

    # Locate the label column containing our tokens
    label_col = None
    # Prefer columns that clearly look like labels
    preferred = ("system", "pipeline", "name", "model")
    # First try to find any column with our token
    for col in result.columns:
        if result[col].dtype == object and result[col].astype(str).str.contains(r"\|size=\d+\|").any():
            label_col = col
            break
    # Heuristic fallback
    if label_col is None:
        for col in preferred:
            if col in result.columns and result[col].astype(str).str.contains(r"\|size=\d+\|").any():
                label_col = col
                break
    if label_col is None:
        raise RuntimeError("Could not locate the pipeline label column containing the '|size=...|' token.")

    # --- Parse tokens from the label into numeric columns ---
    # Size in bytes (int)
    result["disk_size_bytes"] = (
        result[label_col]
        .astype(str)
        .str.extract(r"\|size=(\d+)\|", expand=False)
        .astype("int64")
    )
    result["disk_size_mb"] = result["disk_size_bytes"] / (1024.0 ** 2)

    # k1 and b (floats)
    # Accept integers or decimal floats
    float_pat = r"([0-9]*\.?[0-9]+)"
    result["bm25_k1"] = (
        result[label_col]
        .astype(str)
        .str.extract(rf"\|k1={float_pat}\|", expand=False)
        .astype("float64")
    )
    result["bm25_b"] = (
        result[label_col]
        .astype(str)
        .str.extract(rf"\|b={float_pat}\|", expand=False)
        .astype("float64")
    )

    # Clean the human-readable label (drop the tokens)
    result[label_col] = (
        result[label_col]
        .str.replace(r"\s*\|k1=[^|]*\|\s*", " ", regex=True)
        .str.replace(r"\s*\|b=[^|]*\|\s*", " ", regex=True)
        .str.replace(r"\s*\|size=\d+\|\s*", " ", regex=True)
        .str.replace(r"\s*\(\d+(\.\d+)?\s*MB\)\s*", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Persist results
    result.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
