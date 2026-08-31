import pandas as pd
import pyterrier as pt
from ir_measures import nDCG
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.base import RunConfig, Suite
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

CQADUPSTACK_PREFIX = "beir/cqadupstack/"
CQADUPSTACK_NAME = "beir/cqadupstack"


def document_filter(row):
    """Reject a result row whose document is the query itself."""
    return row.qid != row.docno


def dataframe_filter(df):
    """Drop self-retrieved documents from a result frame."""
    return df[df.apply(document_filter, axis=1)]


class _BEIR(Suite):
    """
    BEIR suite for evaluating retrieval systems on various datasets.

    This suite includes a wide range of datasets from the BEIR benchmark,
    covering domains like question answering, fact verification, and more.
    It uses nDCG@10 as the primary measure for evaluation.

    Example:
        from suiteeval.suite import BEIR
        results = BEIR(pipeline)
    """

    _datasets = datasets
    _measures = measures
    _metadata = {
        "official_measures": measures,
        "description": " Beir is a suite of benchmarks to test zero-shot transfer.",
    }
    _query_field = "text"

    def wrap_pipeline(
        self, pipeline: Transformer, context: DatasetContext
    ) -> Transformer:
        """Filter self-retrieved documents on Quora, where queries are also documents."""
        if "quora" not in context.dataset._irds_id.lower():
            return pipeline
        return pipeline >> pt.apply.generic(dataframe_filter)

    def postprocess_results(
        self, results: pd.DataFrame, config: RunConfig
    ) -> pd.DataFrame:
        """
        Collapse the CQADupStack sub-datasets into a single row before aggregating.

        CQADupStack counts as one BEIR dataset, so its twelve sub-collections are
        reduced to their geometric mean and reported under ``beir/cqadupstack``.
        """
        if results.empty:
            return results

        is_cqadupstack = results["dataset"].str.startswith(CQADUPSTACK_PREFIX)
        if is_cqadupstack.any():
            grouping = ["name", "qid"] if config.perquery else ["name"]
            aggregated = (
                results[is_cqadupstack]
                .groupby(grouping)
                .agg({col: geometric_mean for col in self.metric_columns(results)})
                .reset_index()
            )
            aggregated["dataset"] = CQADUPSTACK_NAME
            results = pd.concat(
                [results[~is_cqadupstack], aggregated], ignore_index=True
            )

        return super().postprocess_results(results, config)


BEIR = _BEIR()

__all__ = ["BEIR"]
