from abc import ABC
from functools import cached_property
from typing import Dict, Generator, List, Optional
import ir_datasets as irds
import pandas as pd
import pyterrier as pt


class SuiteMeta(type):
    """
    Metaclass for Suite:

    - keeps a registry of suite-classes by name
    - enforces singleton instances per suite-class
    - provides a .register(...) helper
    """

    _classes: Dict[str, type] = {}
    _instances: Dict[str, "Suite"] = {}

    def __call__(cls, *args, **kwargs):
        # singleton: only one instance per class
        if cls.__name__ not in SuiteMeta._instances:
            SuiteMeta._instances[cls.__name__] = super().__call__(*args, **kwargs)
        return SuiteMeta._instances[cls.__name__]

    @classmethod
    def register(
        mcs, suite_name: str, datasets: List[str], names: Optional[List[str]] = None
    ) -> "Suite":
        """
        Create (or retrieve) a Suite singleton that wraps the given datasets.

        Args:
            suite_name:  Name of the suite class/instance.
            datasets:    List of ir_datasets identifiers.
            names:       Optional list of dataset-display names;
                         if omitted, uses the same strings as `datasets`.

        Returns:
            The singleton Suite instance.
        """
        # if already registered, return existing instance
        if suite_name in mcs._classes:
            return mcs._classes[suite_name]()

        # build the mapping name→dataset_id
        ds_names = names or datasets
        mapping = dict(zip(ds_names, datasets))

        # dynamically create a new subclass of Suite with the right _datasets
        attrs = {"_datasets": mapping}
        new_cls = mcs(suite_name, (Suite,), attrs)

        # store class and return its singleton instance
        mcs._classes[suite_name] = new_cls
        return new_cls()


class Suite(ABC, metaclass=SuiteMeta):
    """
    Abstract base class for a suite of evaluations.
    Subclasses (and dynamic registrations) must populate:
        _datasets: Dict[str, ir_datasets.Dataset ID]
    """

    _datasets: Dict[str, str] = {}

    @staticmethod
    def get_topics(dataset) -> pd.DataFrame:
        topics = pd.DataFrame(dataset.queries_iter()).rename(
            columns={"query_id": "qid", "text": "query"}
        )
        return topics

    @staticmethod
    def get_qrels(dataset) -> pd.DataFrame:
        qrels = pd.DataFrame(dataset.qrels_iter()).rename(
            columns={"query_id": "qid", "document_id": "docno", "relevance": "label"}
        )
        return qrels

    @cached_property
    def datasets(self) -> Generator[str, irds.Dataset, None]:
        for name, ds_id in self._datasets.items():
            yield name, irds.load(ds_id)

    def __call__(
        self,
        pipelines,
        eval_metrics,
        names=None,
        perquery=False,
        dataframe=True,
        batch_size=None,
        filter_by_qrels=False,
        filter_by_topics=True,
        baseline=None,
        test="t",
        correction=None,
        correction_alpha=0.05,
        highlight=None,
        round=None,
        verbose=False,
        save_dir=None,
        save_mode="warn",
        save_format="trec",
        precompute_prefix=True,
        **kwargs,
    ) -> pd.DataFrame:
        results = []
        for ds_name, ds in self.datasets:
            topics = self.get_topics(ds)
            qrels = self.get_qrels(ds)
            df = self.evaluate(
                pipelines=pipelines,
                eval_metrics=eval_metrics,
                topics=topics,
                qrels=qrels,
                names=names,
                perquery=perquery,
                dataframe=dataframe,
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
                **kwargs,
            )
            df["dataset"] = ds_name
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def evaluate(
        self,
        pipelines,
        eval_metrics,
        topics,
        qrels,
        names=None,
        perquery=False,
        dataframe=True,
        batch_size=None,
        filter_by_qrels=False,
        filter_by_topics=True,
        baseline=None,
        test="t",
        correction=None,
        correction_alpha=0.05,
        highlight=None,
        round=None,
        verbose=False,
        save_dir=None,
        save_mode="warn",
        save_format="trec",
        precompute_prefix=False,
        **kwargs,
    ) -> pd.DataFrame:
        return pt.Experiment(
            retr_systems=pipelines,
            topics=topics,
            qrels=qrels,
            eval_metrics=eval_metrics,
            names=names,
            perquery=perquery,
            dataframe=dataframe,
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
            **kwargs,
        )


"""
# ——— Example usage ———

# Register a new suite (first call creates it; subsequent calls return the same instance)
my_suite = Suite.register(
    suite_name="MyPassageSuite",
    datasets=[
        "msmarco-passage",
        "trec-core17"
    ],
    names=[
        "MSMARCO",
        "Core17"
    ]
)

# Now `my_suite` is a singleton instance wrapping those two collections.
# You can call it exactly like pt.Experiment, plus pipelines & eval_metrics:
# results_df = my_suite(
#     pipelines=[bm25, dpr],
#     eval_metrics=["map", "ndcg"]
# )

"""
