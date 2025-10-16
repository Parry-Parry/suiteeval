from pathlib import Path
import os
from typing import Optional, Union

import click
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_alpha as pta
from pyterrier_caching import ScorerCache, RetrieverCache
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


def rr_linear_fusion(
    r1: pd.DataFrame,
    r2: pd.DataFrame,
    k: int = 60,
    alpha: float = 0.5,
    num_results: Optional[int] = 1000
) -> pd.DataFrame:
    """
    Reciprocal-Rank Linear Interpolation between exactly two ranking result frames.

    Fused score s(d) = alpha * (1/(rank_1(d)+k)) + (1-alpha) * (1/(rank_2(d)+k)).
    Missing documents in a list default to 0 contribution from that list.

    Args:
      r1, r2: result frames with columns ['qid','docno','score'] (and optionally 'query').
      k: constant for reciprocal rank.
      alpha: interpolation weight in [0,1] applied to r1; (1-alpha) applied to r2.
      num_results: number of results to keep per query; if None, keep all.
    """
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

    # Validate and ensure ranks exist
    pta.validate.result_frame(r1, extra_columns=['score'])
    pta.validate.result_frame(r2, extra_columns=['score'])
    pt.model.add_ranks(r1)
    pt.model.add_ranks(r2)

    # Keep optional 'query' if provided; otherwise synthesize a neutral column for merge
    has_query = 'query' in r1.columns and 'query' in r2.columns
    merge_keys = ['qid', 'docno'] + (['query'] if has_query else [])

    # Convert to RRF scores
    s1 = r1[merge_keys + ['rank']].copy()
    s1['rrf1'] = 1.0 / (s1['rank'] + k)
    s1 = s1.drop(columns=['rank'])

    s2 = r2[merge_keys + ['rank']].copy()
    s2['rrf2'] = 1.0 / (s2['rank'] + k)
    s2 = s2.drop(columns=['rank'])

    # Outer merge and interpolate
    merged = s1.merge(s2, how='outer', on=merge_keys)
    merged['rrf1'] = merged['rrf1'].fillna(0.0)
    merged['rrf2'] = merged['rrf2'].fillna(0.0)
    merged = merged.assign(score=alpha * merged['rrf1'] + (1.0 - alpha) * merged['rrf2'])
    merged = merged.drop(columns=['rrf1', 'rrf2'])

    # Rank within query and apply cutoff if requested
    pt.model.add_ranks(merged)
    merged = merged.sort_values(['qid', 'rank'], ascending=[True, True])

    if num_results is not None:
        merged = merged[merged['rank'] < num_results]

    # Reorder columns to match PyTerrier expectations
    cols = ['qid', 'docno', 'score', 'rank']
    if has_query:
        cols = ['qid', 'query', 'docno', 'score', 'rank']
    return merged[cols].reset_index(drop=True)


class RRLinearFusion(pt.Transformer):
    """
    Reciprocal-Rank Linear Interpolation between exactly two transformers.

    For each ranking i ∈ {1,2}, we compute an RRF score s_i(d) = 1 / (rank_i(d) + k).
    The fused score is:  s(d) = alpha * s_1(d) + (1 - alpha) * s_2(d).

    Args:
      transformers: exactly two transformers to fuse.
      k: constant for reciprocal-rank computation.
      alpha: interpolation weight in [0, 1]; weight for the first transformer.
      num_results: number of results to keep per query; if None, keep all.
    """
    schematic = {'inner_pipelines_mode': 'combine'}

    def __init__(
        self,
        *transformers: pt.Transformer,
        k: int = 60,
        alpha: float = 0.5,
        num_results: Optional[int] = 1000
    ):
        assert len(transformers) == 2, "RRLinearFusion requires exactly two transformers"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.transformers = transformers
        self.k = k
        self.alpha = alpha
        self.num_results = num_results

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        r1 = self.transformers[0](inp)
        r2 = self.transformers[1](inp)
        return rr_linear_fusion(r1, r2, k=self.k, alpha=self.alpha, num_results=self.num_results)


@click.command()
@click.option("--save-path", type=str, default='results.csv.gz', help="Path to save the CSV results.")
@click.option("--checkpoint", type=str, default="Shitao/RetroMAE_MSMARCO_finetune", help="Checkpoint for biencoder.")
@click.option("--score-cache-dir", type=str, default="score_cache",
              help="Directory where scorer caches are stored (one subdir per dataset/checkpoint).")
def main(
        save_path: str,
        checkpoint: str,
        score_cache_dir: str,
        ):
    def pipelines(context: DatasetContext):
        # --- cache roots and tags ---
        dataset_tag = Path(context.path).name
        checkpoint_tag = Path(checkpoint).name.replace(os.sep, "_")
        cache_root = Path(score_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        # scorer cache for the bi-encoder re-ranker
        biencoder_cache_dir = cache_root / f"{dataset_tag}__{checkpoint_tag}"

        # retriever caches (cache full retrieved lists)
        retrievers_root = cache_root / "retrievers"
        retrievers_root.mkdir(parents=True, exist_ok=True)
        bm25_cache_path = retrievers_root / f"{dataset_tag}__bm25.dbm"
        e2e_cache_path = retrievers_root / f"{dataset_tag}__{checkpoint_tag}__e2e.dbm"

        # --- index paths ---
        biencoder_dir = f"{context.path}/index.flex"
        pisa_dir = f"{context.path}/index.pisa"

        # --- biencoder indexing ---
        flex_index = FlexIndex(biencoder_dir)
        biencoder = HgfBiEncoder.from_pretrained(checkpoint, batch_size=512)
        indexer_pipe = biencoder >> flex_index
        indexer_pipe.index(context.get_corpus_iter())

        # --- end-to-end retriever, reuse existing cache if found ---
        if e2e_cache_path.exists():
            e2e_retr = RetrieverCache(str(e2e_cache_path))
        else:
            e2e_retr = RetrieverCache(str(e2e_cache_path), flex_index.torch_retriever(), on="query")
        e2e_pipe = biencoder >> e2e_retr

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

        # --- BM25 retriever, reuse existing cache if found ---
        if bm25_cache_path.exists():
            bm25 = RetrieverCache(str(bm25_cache_path))
        else:
            bm25 = RetrieverCache(str(bm25_cache_path), pisa_index.bm25(), on="query")

        # --- bi-encoder scorer cache, reuse existing if found ---
        biencoder_scorer = context.text_loader() >> biencoder
        if biencoder_cache_dir.exists():
            cached_biencoder_scorer = ScorerCache(str(biencoder_cache_dir))
        else:
            cached_biencoder_scorer = ScorerCache(str(biencoder_cache_dir), biencoder_scorer)

        # re-ranking pipeline
        biencoder_pipe = bm25 >> cached_biencoder_scorer

        # interpolation grid
        alphas = [x / 10.0 for x in range(0, 11)]

        for i in alphas:
            yield (
                RRLinearFusion(e2e_pipe, biencoder_pipe, alpha=i),
                f"RRLinearFusion(Bi-Encoder E2E, BM25 >> Bi-Encoder, alpha={i:.1f}) "
                f"|size={pisa_size_b + biencoder_size_b}| ({(pisa_size_mb + biencoder_size_mb):.1f} MB)"
            )

        for i in alphas:
            yield (
                RRLinearFusion(bm25, biencoder_pipe, alpha=i),
                f"RRLinearFusion(BM25, BM25 >> Bi-Encoder, alpha={i:.1f}) "
                f"|size={pisa_size_b + biencoder_size_b}| ({(pisa_size_mb + biencoder_size_mb):.1f} MB)"
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
