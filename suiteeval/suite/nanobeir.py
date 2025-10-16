import builtins
from collections.abc import Sequence as runtime_Sequence
from typing import Any, Sequence, Optional, Union, Tuple

from ir_measures import nDCG
import pandas as pd
import pyterrier as pt
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.base import Suite
from suiteeval.suite.beir import dataframe_filter
from suiteeval.utility import geometric_mean


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


class _NanoBEIR(Suite):
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

    def coerce_pipelines_sequential(
        self,
        context: DatasetContext,
        pipeline_generators: "runtime_Sequence|builtins.callable",
    ):
        """
        Wrap each streamed pipeline with a dataframe filter only for Quora,
        preserving (pipeline, name) pairs and not materialising the sequence.
        """
        ds_str = context.dataset._irds_id.lower()

        for p, nm in super().coerce_pipelines_sequential(context, pipeline_generators):
            if "quora" in ds_str:
                # Append the filter as a no-op transformer for other outputs
                p = p >> pt.apply.generic(dataframe_filter, transform_outputs=lambda x: x)
            yield p, nm

    def coerce_pipelines_grouped(
        self,
        context: DatasetContext,
        pipeline_generators: "runtime_Sequence|builtins.callable",
    ) -> Tuple[list[Transformer], Optional[list[str]]]:
        """
        Materialise all pipelines (and names) via the superclass, then
        append a dataframe filter only for Quora datasets.
        """
        pipelines, names = super().coerce_pipelines_grouped(context, pipeline_generators)

        ds_str = context.dataset._irds_id.lower()

        if "quora" in ds_str:
            pipelines = [
                p >> pt.apply.generic(dataframe_filter, transform_outputs=lambda x: x)
                for p in pipelines
            ]

        return pipelines, names

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
            ranking_generators=pipelines,
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

        if not results:
            return pd.DataFrame()

        quora = results[results["dataset"] == "nano-beir/quora"]
        not_quora = results[results["dataset"] != "nano-beir/quora"]

        quora = quora[quora["qid"] != quora["docno"]]
        results = pd.concat([not_quora, quora], ignore_index=True)

        if not perquery:
            gmean_rows = []
            for (dataset, name), group in results.groupby(["dataset", "name"]):
                row = {"dataset": dataset, "name": name}
                for measure in self._measures:
                    if measure in group:
                        values = group[measure].values
                        gmean = geometric_mean(values)
                        row[measure] = gmean
                gmean_rows.append(row)
            gmean_df = pd.DataFrame(gmean_rows)
            gmean_df["Dataset"] = "Overall"
            results = pd.concat([results, gmean_df], ignore_index=True)

        return results


NanoBEIR = _NanoBEIR()

__all__ = ["NanoBEIR"]
