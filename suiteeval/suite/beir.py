import builtins
from collections.abc import Sequence as runtime_Sequence
from typing import Any, Sequence, Optional, Union, Tuple

from ir_measures import nDCG
import pandas as pd
import pyterrier as pt
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.base import Suite
from suiteeval.utility import geometric_mean

datasets = [
    "beir/arguana",
    "beir/climate-fever",
    "beir/cqadupstack/android",
    "beir/cqadupstack/english",
    "beir/cqadupstack/gaming",
    "beir/cqadupstack/gis",
    "beir/cqadupstack/mathematica",
    "beir/cqadupstack/physics",
    "beir/cqadupstack/programmers",
    "beir/cqadupstack/stats",
    "beir/cqadupstack/tex",
    "beir/cqadupstack/unix",
    "beir/cqadupstack/webmasters",
    "beir/cqadupstack/wordpress",
    "beir/dbpedia-entity/test",
    "beir/fever/test",
    "beir/fiqa/test",
    "beir/hotpotqa/test",
    "beir/msmarco/test",
    "beir/nfcorpus/test",
    "beir/nq",
    "beir/quora/test",
    "beir/scifact/test",
    "beir/trec-covid",
    "beir/webis-touche2020/v2",
]
measures = [nDCG @ 10]


def document_filter(row):
    if row.qid == row.docno:
        return False
    return True


def dataframe_filter(df):
    return df[df.apply(document_filter, axis=1)]


class _BEIR(Suite):
    """
    BEIR suite for evaluating retrieval systems on various datasets.

    This suite includes a wide range of datasets from the BEIR benchmark,
    covering domains like question answering, fact verification, and more.
    It uses nDCG@10 as the primary measure for evaluation.

    Example:
        from suiteeval.suite import BEIR
        beir_suite = BEIR()
        results = beir_suite(pipeline)
    """

    _datasets = datasets
    _measures = measures
    _metadata = {
        "official_measures": measures,
        "description": " Beir is a suite of benchmarks to test zero-shot transfer.",
    }
    _query_field = "text"

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
                p = p >> pt.apply.generic(dataframe_filter)
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
        pipelines, names = super().coerce_pipelines_grouped(
            context, pipeline_generators
        )

        ds_str = context.dataset._irds_id.lower()

        if "quora" in ds_str:
            pipelines = [p >> pt.apply.generic(dataframe_filter) for p in pipelines]

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
        index_dir: Optional[str] = None,
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
            index_dir=index_dir,
        )

        if results is None or results.empty:
            return pd.DataFrame()

        cqadupstack = results[results["dataset"].str.startswith("beir/cqadupstack/")]
        not_cqadupstack = results[
            ~results["dataset"].str.startswith("beir/cqadupstack/")
        ]

        # Group by name (and qid if perquery) to aggregate across cqadupstack sub-datasets
        grouping = ["name"]
        if perquery:
            grouping.append("qid")

        # Determine which metric columns to aggregate (exclude non-metric columns)
        metric_cols = [
            col
            for col in cqadupstack.columns
            if col not in grouping + ["dataset", "name", "qid"]
        ]
        agg_dict = {col: geometric_mean for col in metric_cols}

        cqadupstack = cqadupstack.groupby(grouping).agg(agg_dict).reset_index()
        cqadupstack["dataset"] = "beir/cqadupstack"
        results = pd.concat([not_cqadupstack, cqadupstack], ignore_index=True)

        if not perquery:
            results = self.compute_overall_mean(results)

        return results


BEIR = _BEIR()

__all__ = ["BEIR"]
