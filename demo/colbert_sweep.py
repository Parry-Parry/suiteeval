from pathlib import Path
import os

import click
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory
from pyterrier_pisa import PisaIndex

from suiteeval.context import DatasetContext
from suiteeval import BEIR


def _dir_size_bytes(path: str | os.PathLike) -> int:
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
@click.option("--checkpoint", type=str, default="bert-base-uncased", help="Checkpoint for ColBERT.")
@click.option("--batch-size", type=int, default=512, help="Batch size for processing.")
def main(
        save_path: str,
        checkpoint: str = "bert-base-uncased",
        batch_size: int = 512,
        ):
    def pipelines(context: DatasetContext):
        # Paths
        colbert_dir = f"{context.path}/colbert_index"
        pisa_dir = f"{context.path}/index.pisa"

        # --- ColBERT indexing ---
        colbert_indexer = ColBERTIndexer(checkpoint, colbert_dir, "colbert")
        colbert_indexer.index(context.get_corpus_iter())
        del colbert_indexer  # free memory before ranking

        # Compute on-disk size for ColBERT index
        colbert_size_b = _dir_size_bytes(colbert_dir)
        colbert_size_mb = _mb(colbert_size_b)

        # ColBERT end-to-end pipeline
        colbert = ColBERTFactory(checkpoint, colbert_dir, "colbert")
        yield (
            colbert.end_to_end(batch_size=batch_size),
            f"ColBERT end-to-end |size={colbert_size_b}| ({colbert_size_mb:.1f} MB)"
        )

        # --- PISA (BM25) indexing ---
        pisa_index = PisaIndex(pisa_dir, stemmer="none")
        pisa_index.index(context.get_corpus_iter())

        # Compute on-disk size for PISA index
        pisa_size_b = _dir_size_bytes(pisa_dir)
        pisa_size_mb = _mb(pisa_size_b)

        # BM25 >> ColBERT rescoring pipeline
        yield (
            pisa_index.bm25() >> context.text_loader() >> colbert.text_scorer(batch_size=batch_size),
            f"BM25 >> ColBERT |size={pisa_size_b}| ({pisa_size_mb:.1f} MB)"
        )

    result = BEIR(pipelines)

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
