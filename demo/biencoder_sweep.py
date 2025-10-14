from pathlib import Path
import os
from typing import Union

import click
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_dr import HgfBiEncoder, FlexIndex
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
@click.option("--save-path", type=str, required=True, help="Path to save the CSV results.")
@click.option("--checkpoint", type=str, default="bert-base-uncased", help="Checkpoint for biencoder.")
def main(
        save_path: str,
        checkpoint: str = "bert-base-uncased",
        ):
    def pipelines(context: DatasetContext):
        # Paths
        biencoder_dir = f"{context.path}/index.flex"
        pisa_dir = f"{context.path}/index.pisa"

        # --- biencoder indexing ---
        flex_index = FlexIndex(biencoder_dir)
        biencoder = HgfBiEncoder.from_pretrained(checkpoint, batch_size=512)
        e2e_pipe = biencoder >> flex_index
        e2e_pipe.index(context.get_corpus_iter())

        # Compute on-disk size for biencoder index
        biencoder_size_b = _dir_size_bytes(biencoder_dir)
        biencoder_size_mb = _mb(biencoder_size_b)

        yield (
            e2e_pipe,
            f"biencoder end-to-end |size={biencoder_size_b}| ({biencoder_size_mb:.1f} MB)"
        )

        # --- PISA (BM25) indexing ---
        pisa_index = PisaIndex(pisa_dir, stemmer="none")
        pisa_index.index(context.get_corpus_iter())

        # Compute on-disk size for PISA index
        pisa_size_b = _dir_size_bytes(pisa_dir)
        pisa_size_mb = _mb(pisa_size_b)

        # BM25 >> biencoder rescoring pipeline
        yield (
            pisa_index.bm25() >> context.text_loader() >> biencoder,
            f"BM25 >> biencoder |size={pisa_size_b}| ({pisa_size_mb:.1f} MB)"
        )

    result = BEIR()(pipelines)

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
