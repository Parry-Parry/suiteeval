from typing import Any, Sequence, Optional, Union, Dict

from ir_measures import nDCG
from suiteeval.suite.base import Suite
import pandas as pd

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
        round: Optional[Union[int, Dict[str, int]]] = None,
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

        cqadupstack = results[results["dataset"].str.startswith("beir/cqadupstack/")]
        not_cqadupstack = results[
            ~results["dataset"].str.startswith("beir/cqadupstack/")
        ]

        if perquery:
            cqadupstack = (
                cqadupstack.groupby(["dataset", "qid", "name"]).mean().reset_index()
            )
        else:
            cqadupstack = cqadupstack.groupby(["dataset", "name"]).mean().reset_index()
        cqadupstack["dataset"] = "beir/cqadupstack"
        results = pd.concat([not_cqadupstack, cqadupstack], ignore_index=True)

        quora = results[results["dataset"] == "beir/quora/test"]
        not_quora = results[results["dataset"] != "beir/quora/test"]

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


BEIR = _BEIR()

__all__ = ["BEIR"]
