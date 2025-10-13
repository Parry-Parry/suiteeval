from abc import ABCMeta, ABC
from functools import cache
from typing import Dict, Generator, List, Optional, Any, Tuple, Union, Sequence
import builtins
import inspect
from collections.abc import Sequence as runtime_Sequence, Iterator
from logging import getLogger

import ir_datasets as irds
from ir_measures import nDCG, Measure, parse_measure, parse_trec_measure
import pandas as pd
import pyterrier as pt
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.utility import geometric_mean

logging = getLogger(__name__)


class SuiteMeta(ABCMeta):
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
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
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

        # build the dataset name → dataset_id mapping
        ds_names    = names or datasets
        dataset_map = dict(zip(ds_names, datasets))

        # normalize metadata: could be
        #  • None            → no per‐dataset metadata
        #  • list[dict]      → metadata[i] applies to ds_names[i]
        #  • dict[str,dict]  → per‐dataset mapping (keys are names or IDs)
        #  • dict[k,v] where v is NOT a dict → flat metadata for all
        if metadata is None:
            metadata_map = {name: {} for name in ds_names}

        elif isinstance(metadata, list):
            if len(metadata) != len(ds_names):
                raise ValueError("`metadata` list must match number of datasets")
            metadata_map = dict(zip(ds_names, metadata))

        elif isinstance(metadata, dict):
            # check if values are all non‐dict → treat as global metadata
            if all(not isinstance(v, dict) for v in metadata.values()):
                metadata_map = {name: metadata for name in ds_names}
            else:
                # assume user passed a per‐dataset dict
                metadata_map = metadata

        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

        # now dynamically create the subclass with both mappings
        attrs = {
            "_datasets": dataset_map,
            "_metadata": metadata_map
        }
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
            if not len(self._measures) == 0:
                return

        if any(ds_id in metadata for ds_id in self._datasets):
            # If any dataset in _datasets has metadata, use that
            self._measures = {}
            for ds_id, ds_metadata in metadata.items():
                if ds_id in self._datasets:
                    self._measures[ds_id] = self.parse_measures(
                        ds_metadata.get("official_measures", self.__default_measures)
                    )
            if all(
                isinstance(m, Measure) for m in self._measures.values()
            ):
                # If all measures are valid, return
                return

        for ds_id, ds in self._datasets.items():
            if self._measures is None:
                self._measures = {}
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

        if any(not isinstance(m, Measure) for m in self._measures.values()):
            for ds_id, measures in self._measures.items():
                # if empty list or not at all default to nDCG@10 and warn
                if not measures or not isinstance(measures, list):
                    logging.warning(
                        f"Dataset {ds_id} has no valid measures defined. Defaulting to nDCG@10."
                    )
                    self._measures[ds_id] = self.__default_measures

        if self._measures is None:
            logging.warning(
                "No measures defined for this suite. Defaulting to nDCG@10."
            )
            self._measures = [nDCG @ 10]

    def coerce_pipelines(
                        self,
                        context: DatasetContext,
                        pipeline_generators: "runtime_Sequence|builtins.callable",
                    ) -> Tuple[List[Transformer], Optional[List[str]]]:
        """
        Coerces indexing and ranking generators to pipelines.
        """
        if not isinstance(pipeline_generators, runtime_Sequence) or isinstance(pipeline_generators, (str, bytes)):
            if not builtins.callable(pipeline_generators):
                raise TypeError("pipeline_generators must be a callable or a sequence of callables.")
            gens = [pipeline_generators]
        else:
            if not all(builtins.callable(f) for f in pipeline_generators):
                raise TypeError("All elements of pipeline_generators must be callable.")
            gens = list(pipeline_generators)
        pipelines: List[Transformer] = []
        names: List[str] = []

        for gen in gens:
            out = gen(context)  # pass DatasetContext, not corpus iterator

            # Case 1: generator/iterator of (pipeline, name) pairs
            if inspect.isgenerator(out) or isinstance(out, Iterator):
                pairs = list(out)
                for item in pairs:
                    if isinstance(item, tuple) and len(item) == 2:
                        p, nm = item
                    else:
                        # allow yielding a bare Transformer (name becomes None)
                        p, nm = item, None
                    if isinstance(p, Transformer):
                        pipelines.append(p)
                        names.append(nm if isinstance(nm, str) else None)
                    elif isinstance(p, runtime_Sequence) and all(isinstance(pi, Transformer) for pi in p):
                        pipelines.extend(p)
                        # broadcast name if string; else append Nones
                        if isinstance(nm, str):
                            names.extend([nm] * len(p))
                        elif isinstance(nm, runtime_Sequence):
                            names.extend(list(nm))
                        else:
                            names.extend([None] * len(p))
                    else:
                        raise ValueError(
                            f"Pipeline generator {gen} yielded an invalid item: {type(p)}"
                        )
                continue

            # Case 2: return-style — (pipelines, names?) or a single Transformer / sequence
            if isinstance(out, tuple):
                _pipelines, *_rest = out
                _names = None if not _rest else _rest[0]
            else:
                _pipelines, _names = out, None

            # Normalize _pipelines
            if isinstance(_pipelines, Transformer):
                pipelines.append(_pipelines)
                names.append(_names if isinstance(_names, str) else None)
            elif inspect.isgenerator(_pipelines) or isinstance(_pipelines, Iterator):
                _pipelines = list(_pipelines)

            if isinstance(_pipelines, runtime_Sequence) and all(isinstance(pi, Transformer) for pi in _pipelines):
                pipelines.extend(_pipelines)
                if isinstance(_names, str):
                    names.extend([_names] * len(_pipelines))
                elif isinstance(_names, runtime_Sequence):
                    names.extend(list(_names))
                else:
                    names.extend([None] * len(_pipelines))
            elif not isinstance(_pipelines, Transformer):
                raise ValueError(
                    f"Pipeline generator {gen} must return or yield a Transformer "
                    f"or a sequence of Transformers."
                )

        if not pipelines:
            raise ValueError("No pipelines generated. Ensure your generators produce valid Transformers.")

        # Collapse names to None if all missing
        final_names = None if not any(names) else [nm if nm is not None else f"pipeline_{i}" for i, nm in enumerate(names)]
        return pipelines, final_names

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

    @property
    def datasets(self) -> Generator[Tuple[str, pt.datasets.Dataset], None, None]:
        """
        Returns a generator yielding dataset names and their ir_datasets.Dataset instances.
        Yields:
            Tuple of (dataset_name, dataset_instance)
        """
        if type(self._datasets) is list:
            # If _datasets is a list, assume they are dataset IDs
            for ds_id in self._datasets:
                yield ds_id, pt.get_dataset(f"irds:{ds_id}")
        elif type(self._datasets) is dict:
            for name, ds_id in self._datasets.items():
                yield name, pt.get_dataset(f"irds:{ds_id}")
        else:
            raise ValueError(
                "Suite _datasets must be a list or dict mapping names to dataset IDs."
            )

    def __call__(
        self,
        ranking_generators: Union[callable, Sequence[callable]],
        eval_metrics: Sequence[Any] = None,
        subset: Optional[str] = None,
        **experiment_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        results = []
        for ds_name, ds in self.datasets:
            if subset and ds_name != subset:
                continue

            topics = ds.get_topics()
            qrels = ds.get_qrels()
            context = DatasetContext(ds)
            pipelines, names = self.coerce_pipelines(context, ranking_generators)
            df = pt.Experiment(
                pipelines,
                eval_metrics=eval_metrics or self.get_measures(ds_name),
                topics=topics,
                qrels=qrels,
                names=names,
                **experiment_kwargs,
            )
            df["dataset"] = ds_name
            results.append(df)

        results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        perquery = experiment_kwargs.get("perquery", False)
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
