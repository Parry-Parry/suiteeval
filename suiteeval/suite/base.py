from abc import ABC
from functools import cache, cached_property
from typing import Dict, Generator, List, Optional, Any, Union, Sequence
import ir_datasets as irds
from ir_measures import nDCG, Measure, parse_measure, parse_trec_measure
import pandas as pd
import pyterrier as pt
from pyterrier import Transformer
from logging import getLogger

from suiteeval.context import DatasetContext

logging = getLogger(__name__)


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
        mcs,
        suite_name: str,
        datasets: List[str],
        names: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "Suite":
        """
        Create (or retrieve) a Suite singleton that wraps the given datasets.

        Args:
            suite_name:  Name of the suite class/instance.
            datasets:    List of ir_datasets identifiers.
            names:       Optional list of dataset-display names;
                         if omitted, uses the same strings as `datasets`.
            metadata:    Optional list of metadata dictionaries for each dataset.

        Returns:
            The singleton Suite instance.
        """
        # if already registered, return existing instance
        if suite_name in mcs._classes:
            return mcs._classes[suite_name]()

        # build the mapping name→dataset_id
        ds_names = names or datasets
        mapping = dict(zip(ds_names, datasets, metadata or [{}] * len(datasets)))

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

    _datasets: Union[List[str], Dict[str, str]] = {}
    _metadata: Dict[str, Any] = {}
    _measures: List[Measure] = None
    __default_measures: List[Measure] = [nDCG @ 10]

    def __init__(self):
        """
        Initializes the suite.
        """
        self.coerce_measures(self._metadata)
        if "description" in self._metadata:
            self.__doc__ = self._metadata["description"]
        self.__post_init__()

    def __post_init__(self):
        assert (
            self._datasets
        ), "Suite must have at least one dataset defined in _datasets"
        assert isinstance(self._datasets, dict) or isinstance(
            self._datasets, list
        ), "Suite _datasets must be a dict mapping names to dataset ID or a list of dataset IDs"
        if isinstance(self._datasets, dict):
            assert all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in self._datasets.items()
            ), "Suite _datasets must map string names to string dataset IDs"
        else:
            assert all(
                isinstance(ds, str) for ds in self._datasets
            ), "Suite _datasets must be a list of dataset IDs"
        assert (
            self._measures is not None
        ), "Suite must have measures defined in _measures"

    @staticmethod
    def parse_measures(measures: List[Union[str, Measure]]) -> List[Measure]:
        """
        Parses a list of measures, converting strings to Measure instances.
        """
        parsed_measures = []
        for measure in measures:
            if isinstance(measure, str):
                # Convert string to Measure instance
                try:
                    parsed_measure = parse_measure(measure)
                except ValueError:
                    pass
                try:
                    parsed_measure = parse_trec_measure(measure)
                except ValueError:
                    pass

                if type(parsed_measure) is Measure:
                    parsed_measure = [parsed_measure]

                for pm in parsed_measure:
                    if isinstance(pm, Measure):
                        parsed_measures.append(pm)
                    else:
                        raise ValueError(f"Invalid measure type: {type(pm)}")
            elif isinstance(measure, Measure):
                parsed_measures.append(measure)
            else:
                raise ValueError(f"Invalid measure type: {type(measure)}")
        return parsed_measures

    def coerce_measures(self, metadata: List[Dict[str, Any]]) -> None:
        """
        Checks for recommended measures in the metadata or optionally the dataset documentation.
        """
        if "official_measures" in metadata:
            # If the metadata has official measures, use those
            self._measures = self.parse_measures(metadata["official_measures"])
            return

        if any(ds_id in metadata for ds_id in self._datasets):
            # If any dataset in _datasets has metadata, use that
            for ds_id, ds_metadata in metadata.items():
                if ds_id in self._datasets:
                    self._measures[ds_id] = self.parse_measures(
                        ds_metadata.get("official_measures", [])
                    )
            return

        for ds_id, ds in self._datasets.items():
            try:
                # Try to load the dataset and its measures
                dataset = irds.load(ds)
                documentation = dataset.documentation()
                if hasattr(documentation, "official_measures"):
                    self._measures[ds_id] = self.parse_measures(
                        documentation["official_measures"]
                    )

            except Exception as e:
                logging.warning(f"Failed to load measures for dataset {ds_id}: {e}")
                pass
        if self._measures is None:
            logging.warning(
                "No measures defined for this suite. Defaulting to nDCG@10."
            )
            self._measures = [nDCG @ 10]

    def coerce_pipelines(
        self, context: DatasetContext, pipeline_generators: Sequence[callable]
    ) -> Generator[Transformer]:
        """
        Coerces indexing and ranking generators to pipelines.
        """
        pipelines, names = [], []
        for gen in pipeline_generators:
            _pipelines, *args = gen(context.get_corpus_iter())
            _names = None if len(args) == 0 else args[0]
            if isinstance(_pipelines, Transformer):
                # If the generator yields a single pipeline, yield it directly
                pipelines.append(_pipelines)
                if names:
                    names.append(_names)
            elif isinstance(pipelines, Sequence):
                pipelines.extend(_pipelines)
                if _names:
                    names.extend(_names)
            else:
                raise ValueError(
                    f"Pipeline generator {gen} must yield a Transformer or a sequence of Transformers."
                )
        if not pipelines:
            raise ValueError(
                "No pipelines generated. Ensure your pipeline generators yield valid Transformers."
            )
        names = names or None
        return pipelines, names

    @cache
    def get_measures(self, dataset) -> List[Measure]:
        """
        Returns the measures for the given dataset.
        If the suite has a single set of measures, it returns that.
        """
        if type(self._measures) is list:
            return self._measures
        if dataset not in self._measures:
            return self.__default_measures
        return self._measures[dataset]

    @cache
    def get_topics(self, dataset) -> pd.DataFrame:
        """
        Returns the topics DataFrame for the given dataset.
        Columns:
            - qid: Query ID
            - query: Query text
        """
        topics = pd.DataFrame(dataset.queries_iter()).rename(
            columns={"query_id": "qid", "text": "query"}
        )
        return topics

    @cache
    def get_qrels(self, dataset) -> pd.DataFrame:
        """
        Returns the qrels DataFrame for the given dataset.
        Columns:
            - qid: Query ID
            - docno: Document ID
            - label: Relevance label
        """
        qrels = pd.DataFrame(dataset.qrels_iter()).rename(
            columns={"query_id": "qid", "document_id": "docno", "relevance": "label"}
        )
        return qrels

    @cached_property
    def datasets(self) -> Generator[str, irds.Dataset, None]:
        """
        Returns a generator yielding dataset names and their ir_datasets.Dataset instances.
        Yields:
            Tuple of (dataset_name, dataset_instance)
        """
        if type(self._datasets) is list:
            # If _datasets is a list, assume they are dataset IDs
            for ds_id in self._datasets:
                yield ds_id, irds.load(ds_id)
        elif type(self._datasets) is dict:
            for name, ds_id in self._datasets.items():
                yield name, irds.load(ds_id)
        else:
            raise ValueError(
                "Suite _datasets must be a list or dict mapping names to dataset IDs."
            )

    def __call__(
        self,
        ranking_generators: Union[callable, Sequence[callable]],
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
        results = []
        for ds_name, ds in self.datasets:
            if subset and ds_name != subset:
                continue

            topics = self.get_topics(ds)
            qrels = self.get_qrels(ds)
            context = DatasetContext(ds)
            pipelines, names = self.coerce_pipelines(ds, ranking_generators)
            df = pt.Experiment(
                pipelines=pipelines,
                eval_metrics=eval_metrics or self.get_measures(ds_name),
                topics=topics,
                qrels=qrels,
                perquery=perquery,
                dataframe=True,
                filter_by_qrels=filter_by_qrels,
                filter_by_topics=filter_by_topics,
                names=names,
                batch_size=batch_size,
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
            df["dataset"] = ds_name
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
