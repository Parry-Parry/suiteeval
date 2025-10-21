from typing import Any, Sequence, Optional, Union

from ir_measures import nDCG
import pandas as pd

datasets = [
    "nano-beir/arguana",
    "nano-beir/climate-fever",
    "nano-beir/dbpedia-entity",
    "nano-beir/fever",
    "nano-beir/fiqa",
    "nano-beir/hotpotqa",
    "nano-beir/msmarco",
    "nano-beir/nfcorpus",
    "nano-beir/nq",
    "nano-beir/quora",
    "nano-beir/scidocs",
    "nano-beir/scifact",
    "nano-beir/webis-touche2020",
]

measures = [nDCG@10]


class _NanoBEIR(_BEIR):
    """
    Nano BEIR suite for evaluating retrieval systems on various datasets.

    This suite includes a subset and subsampling of datasets from the BEIR benchmark,
    covering domains like question answering, fact verification, and more.
    It uses nDCG@10 as the primary measure for evaluation.

    Example:
        from suiteeval.suite import NanoBEIR
        results = NanoBEIR(pipeline)
    """

    _datasets = datasets
    _measures = measures
    metadata = {
        "official_measures": measures,
        "description": "Nano Beir is a smaller version (max 50 queries per benchmark) of the Beir suite of benchmarks to test zero-shot transfer.",
    }

    def __call__(
        self,
        pipelines: Sequence[Any] = None,
        eval_metrics: Sequence[Any] = None,
        subset: Optional[str] = None,
        perquery: bool = False,
        batch_size: Optional[int] = None,
        filter_by_qrels: bool = False,
        filter_by_topics: bool = True,
        baseline: Optional[int] = None,
        test: str = "t",
        correction: Optional[str] = None,
        correction_alpha: float = 0.05,
        highlight: Optional[str] = None,
        round: Optional[Union[int, dict[str, int]]] = None,
        verbose: bool = False,
        save_dir: Optional[str] = None,
        save_mode: str = "warn",
        save_format: str = "trec",
        precompute_prefix: bool = False,
    ) -> pd.DataFrame:
        results = super().__call__(
            pipelines,
            eval_metrics=eval_metrics,
            subset=subset,
            perquery=perquery,
            batch_size=batch_size,
            filter_by_qrels=filter_by_qrels,
            filter_by_topics=filter_by_topics,
            baseline=baseline,
            test=test,
            correction=correction,
            correction_alpha=correction_alpha,
            highlight=highlight,
            round=round,
            verbose=verbose,
            save_dir=save_dir,
            save_mode=save_mode,
            save_format=save_format,
            precompute_prefix=precompute_prefix,
        )

        if results is None or results.empty:
            return pd.DataFrame()

        if not perquery:
            results = self.compute_overall_mean(results)

        return results


NanoBEIR = _NanoBEIR()

__all__ = ["NanoBEIR"]
