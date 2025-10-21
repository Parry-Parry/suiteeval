from pathlib import Path
import os
from typing import Optional, Union

import click
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_alpha as pta
from pyterrier_dr import HgfBiEncoder, FlexIndex
from pyterrier_pisa import PisaIndex

from suiteeval.context import DatasetContext
from suiteeval import NanoBEIR


def _dir_size_bytes(path: Union[str, os.PathLike]) -> int:
    """Return total size in bytes for a file or directory (recursive)."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    # Fast walk using os.scandir
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
                    # Skip entries deleted during traversal
                    continue
    return total


def _mb(x_bytes: int) -> float:
    return x_bytes / (1024.0 ** 2)


@click.command()
@click.option("--save-path", type=str, default='results.csv.gz', help="Path to save the CSV results.")
@click.option("--checkpoint", type=str, default="Shitao/RetroMAE_MSMARCO_finetune", help="Checkpoint for biencoder.")
              help="Directory where scorer caches are stored (one subdir per dataset/checkpoint).")
def main(
        save_path: str,
        checkpoint: str,
        ):
    def pipelines(context: DatasetContext):

        # --- index paths ---
        biencoder_dir = f"{context.path}/index.flex"
        pisa_dir = f"{context.path}/index.pisa"

        # --- biencoder indexing ---
        flex_index = FlexIndex(biencoder_dir)
        biencoder = HgfBiEncoder.from_pretrained(checkpoint, batch_size=512)
        indexer_pipe = biencoder >> flex_index
        indexer_pipe.index(context.get_corpus_iter())

        e2e_pipe = biencoder >> flex_index

        # Compute on-disk size for biencoder index
        biencoder_size_b = _dir_size_bytes(biencoder_dir)
        biencoder_size_mb = _mb(biencoder_size_b)

        yield (
            e2e_pipe,
            f"Bi-Encoder end-to-end |size={biencoder_size_b}| ({biencoder_size_mb:.1f} MB)"
        )

        # --- PISA (BM25) indexing ---
        pisa_index = PisaIndex(pisa_dir, stemmer="none")
        pisa_index.index(context.get_corpus_iter())

        # Compute on-disk size for PISA index
        pisa_size_b = _dir_size_bytes(pisa_dir)
        pisa_size_mb = _mb(pisa_size_b)

        bm25 = pisa_index.bm25()

        biencoder_scorer = context.text_loader() >> biencoder

        # re-ranking pipeline
        biencoder_pipe = bm25 >> biencoder_scorer

        yield (
            biencoder_pipe,
            f"BM25 >> Bi-Encoder |size={pisa_size_b}| ({pisa_size_mb:.1f} MB)"
        )

    result = NanoBEIR(pipelines)

    # Identify the label column that contains our parse marker
    label_col = None
    for col in result.columns:
        if result[col].dtype == object and result[col].astype(str).str.contains(r"\|size=\d+\|").any():
            label_col = col
            break

    if label_col is None:
        # Fallback: common label column names you might be using
        for candidate in ("system", "pipeline", "name", "model"):
            if candidate in result.columns and result[candidate].astype(str).str.contains(r"\|size=\d+\|").any():
                label_col = candidate
                break

    if label_col is None:
        # If still not found, raise an informative error to catch schema changes early
        raise RuntimeError("Could not locate the pipeline label column containing the '|size=...|' token.")

    # Extract bytes as integer
    result["disk_size_bytes"] = (
        result[label_col]
        .astype(str)
        .str.extract(r"\|size=(\d+)\|", expand=False)
        .astype("int64")
    )

    result["disk_size_mb"] = result["disk_size_bytes"] / (1024.0 ** 2)

    result[label_col] = result[label_col].str.replace(r"\s*\|size=\d+\|\s*", " ", regex=True).str.strip()

    result.to_csv(save_path)


if __name__ == "__main__":
    main()
