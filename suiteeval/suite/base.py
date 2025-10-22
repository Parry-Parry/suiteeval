from __future__ import annotations

from abc import ABCMeta, ABC
import builtins
from collections.abc import Sequence as runtime_Sequence, Iterator
import inspect
from functools import cache
from typing import Callable, Generator, Optional, Any, Tuple, Union, Sequence
from logging import getLogger

import numpy as np
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

    _classes: dict[str, type] = {}
    _instances: dict[str, "Suite"] = {}

    def __call__(cls, *args, **kwargs):
        # singleton: only one instance per class
        if cls.__name__ not in SuiteMeta._instances:
            SuiteMeta._instances[cls.__name__] = super().__call__(*args, **kwargs)
        return SuiteMeta._instances[cls.__name__]

    @classmethod
    def register(
        mcs,
        suite_name: str,
        datasets: list[str],
        names: Optional[list[str]] = None,
        metadata: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None,
        query_field: Optional[str] = None,
    ) -> "Suite":
        """
        Create (or retrieve) a Suite singleton that wraps the given datasets.

        Args:
            suite_name:  Name of the suite class/instance.
            datasets:    list of ir_datasets identifiers.
            names:       Optional list of dataset-display names; defaults to `datasets`.
            metadata:    Optional list/dict of metadata for each dataset.
        """
        # if already registered, return existing instance
        if suite_name in mcs._classes:
            return mcs._classes[suite_name]()

        # build the dataset name → dataset_id mapping
        ds_names = names or datasets
        dataset_map = dict(zip(ds_names, datasets))

        # normalise metadata:
        #  • None            → empty per-dataset dicts
        #  • list[dict]      → metadata[i] applies to ds_names[i]
        #  • dict[str,dict]  → per-dataset mapping (keys are names or IDs)
        #  • dict[k,v] where v is NOT a dict → flat metadata for all
        if metadata is None:
            metadata_map = {name: {} for name in ds_names}
        elif isinstance(metadata, list):
            if len(metadata) != len(ds_names):
                raise ValueError("`metadata` list must match number of datasets")
            metadata_map = dict(zip(ds_names, metadata))
        elif isinstance(metadata, dict):
            if all(not isinstance(v, dict) for v in metadata.values()):
                metadata_map = {name: metadata for name in ds_names}
            else:
                metadata_map = metadata
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

        # dynamically create subclass with mappings
        attrs = {
            "_datasets": dataset_map,     # display-name -> dataset_id
            "_dataset_ids": dataset_map,  # alias used by other methods
            "_metadata": metadata_map,
            "_query_field": query_field,
        }
        new_cls = mcs(suite_name, (Suite,), attrs)

        # store class and return its singleton instance
        mcs._classes[suite_name] = new_cls
        return new_cls()


class Suite(ABC, metaclass=SuiteMeta):
    """
    Abstract base class for a suite of evaluations.
    Subclasses (and dynamic registrations) must populate:
        _datasets: dict[str, ir_datasets.Dataset ID]  (or list[str])
    """

    _datasets: Union[list[str], dict[str, str]] = {}
    _dataset_ids: dict[str, str] = {}
    _metadata: dict[str, Any] = {}
    _measures: Union[list[Measure], dict[str, list[Measure]]] = None
    __default_measures: list[Measure] = [nDCG @ 10]
    _query_field: Optional[str] = None

    # ---------------------------
    # Construction and validation
    # ---------------------------
    def __init__(self):
        self.coerce_measures(self._metadata)
        if "description" in self._metadata:
            self.__doc__ = self._metadata["description"]
        self.__post_init__()

    def __post_init__(self):
        assert self._datasets, "Suite must have at least one dataset defined in _datasets"

        if not isinstance(self._datasets, (dict, list)):
            raise AssertionError("Suite _datasets must be a dict[name->id] or a list[dataset_id]")

        if isinstance(self._datasets, dict):
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in self._datasets.items()):
                raise AssertionError("Suite _datasets must map string names to string dataset IDs")
        else:
            if not all(isinstance(ds, str) for ds in self._datasets):
                raise AssertionError("Suite _datasets list must contain dataset IDs (str)")

        assert self._measures is not None, "Suite must have measures defined in _measures"

    # ---------------------------
    # Corpus grouping
    # ---------------------------
    def _iter_corpus_groups(self):
        """
        Yield groups of datasets that share the same underlying corpus, determined by
        ir_datasets.docs_parent_id(dataset_id).

        Yields:
            (corpus_id: str,
             corpus_ds: pt.datasets.Dataset,
             members: list[tuple[str, str]])  # [(display_name, dataset_id), ...]
        """
        # normalise to a list of (name, ds_id)
        if isinstance(self._datasets, dict):
            items = list(self._datasets.items())
        else:
            items = [(ds_id, ds_id) for ds_id in self._datasets]

        # group by docs-parent (corpus) id
        groups: dict[str, dict] = {}
        for name, ds_id in items:
            try:
                corpus_id = irds.docs_parent_id(ds_id) or ds_id
            except Exception:
                corpus_id = ds_id

            if corpus_id not in groups:
                groups[corpus_id] = {
                    "corpus_ds": pt.get_dataset(f"irds:{corpus_id}"),
                    "members": [],
                }
            groups[corpus_id]["members"].append((name, ds_id))

        # deterministic iteration order (insertion order is fine here)
        for corpus_id, g in groups.items():
            yield corpus_id, g["corpus_ds"], g["members"]

    # ---------------------------
    # Measures
    # ---------------------------
    @staticmethod
    def parse_measures(measures: list[Union[str, Measure]]) -> list[Measure]:
        """
        Convert a list of measure strings or Measure objects to a flat list[Measure].
        """
        out: list[Measure] = []

        def _ensure_list(x: Union[Measure, Sequence[Measure]]) -> list[Measure]:
            if isinstance(x, Measure):
                return [x]
            return list(x)

        for m in measures:
            if isinstance(m, Measure):
                out.append(m)
                continue

            if isinstance(m, str):
                candidates: list[Measure] = []
                for parser in (parse_measure, parse_trec_measure):
                    try:
                        parsed = parser(m)
                        candidates.extend(_ensure_list(parsed))
                    except ValueError:
                        continue
                if not candidates:
                    raise ValueError(f"Unrecognised measure string: {m!r}")
                out.extend(candidates)
                continue

            raise ValueError(f"Invalid measure type: {type(m)}")

        return out

    def coerce_measures(self, metadata: dict[str, Any]) -> None:
        """
        Populate self._measures as a de-duplicated list of Measure objects aggregated from:
          1) Global metadata['official_measures']
          2) Per-dataset metadata[name]['official_measures']
          3) ir_datasets documentation['official_measures'] for each dataset
        Fallback: [nDCG@10] if none found.
        """
        measures_accum: list[Measure] = []
        seen: set[str] = set()

        def _add_many(items: Optional[list[Union[str, Measure]]]) -> None:
            if not items:
                return
            for m in self.parse_measures(items):
                sig = str(m)
                if sig not in seen:
                    measures_accum.append(m)
                    seen.add(sig)

        # (1) global metadata
        if isinstance(metadata, dict):
            _add_many(metadata.get("official_measures"))

        # (2) per-dataset metadata
        if isinstance(metadata, dict):
            # iterate over declared dataset names (works for dict; if list, keys are ids)
            names_iter = self._datasets if isinstance(self._datasets, dict) else self._datasets
            for name in names_iter:
                md = metadata.get(name, {})
                if isinstance(md, dict):
                    _add_many(md.get("official_measures"))

        # (3) ir_datasets documentation
        for name, ds_id in (self._dataset_ids.items() if isinstance(self._dataset_ids, dict) else []):
            try:
                ds = irds.load(ds_id)
                docs = getattr(ds, "documentation", lambda: None)()
                if isinstance(docs, dict):
                    _add_many(docs.get("official_measures"))
            except Exception as e:
                logging.warning(f"Failed to load measures from documentation for '{name}' ({ds_id}): {e}")

        if not measures_accum:
            logging.warning("No measures discovered; defaulting to [nDCG@10].")
            measures_accum = [nDCG @ 10]

        self._measures = measures_accum

    # ---------------------------
    # Pipeline coercion helpers
    # ---------------------------
    @staticmethod
    def _normalize_generators(
        pipeline_generators: Union[Callable[[DatasetContext], Any], runtime_Sequence],
        what: str,
    ) -> list[Callable[[DatasetContext], Any]]:
        """
        Normalise a callable or a sequence of callables to a list of callables.
        """
        if not isinstance(pipeline_generators, runtime_Sequence) or isinstance(pipeline_generators, (str, bytes)):
            if not builtins.callable(pipeline_generators):
                raise TypeError(f"{what} must be a callable or a sequence of callables.")
            return [pipeline_generators]  # type: ignore[list-item]
        if not all(builtins.callable(f) for f in pipeline_generators):  # type: ignore[arg-type]
            raise TypeError(f"All elements of {what} must be callable.")
        return list(pipeline_generators)  # type: ignore[return-value]

    def coerce_pipelines_sequential(
        self,
        context: DatasetContext,
        pipeline_generators: Union[Callable[[DatasetContext], Any], runtime_Sequence],
    ):
        """
        Yield (Transformer, Optional[str]) one at a time, without materialising.
        Use to reduce memory/VRAM footprint.
        """
        gens = self._normalize_generators(pipeline_generators, "pipeline_generators")

        def _yield_item(item):
            if isinstance(item, tuple) and len(item) == 2:
                p, nm = item
            else:
                p, nm = item, None

            if isinstance(p, Transformer):
                yield p, (nm if isinstance(nm, str) else None)
            elif isinstance(p, runtime_Sequence) and all(isinstance(pi, Transformer) for pi in p):
                if isinstance(nm, str):
                    for pi in p:
                        yield pi, nm
                elif isinstance(nm, runtime_Sequence):
                    nm_list = list(nm)
                    if len(nm_list) != len(p):
                        raise ValueError("Length of names does not match number of pipelines.")
                    for pi, nmi in zip(p, nm_list):
                        yield pi, (nmi if isinstance(nmi, str) else None)
                else:
                    for pi in p:
                        yield pi, None
            else:
                raise ValueError(f"Generator yielded an invalid item: {type(p)}")

        for gen in gens:
            out = gen(context)
            if inspect.isgenerator(out) or isinstance(out, Iterator):
                for item in out:
                    yield from _yield_item(item)
            else:
                if isinstance(out, tuple):
                    _pipelines, *_rest = out
                    _names = None if not _rest else _rest[0]
                    yield from _yield_item((_pipelines, _names))
                else:
                    yield from _yield_item(out)

    def coerce_pipelines_grouped(
        self,
        context: DatasetContext,
        pipeline_generators: Union[Callable[[DatasetContext], Any], runtime_Sequence],
    ) -> Tuple[list[Transformer], Optional[list[str]]]:
        """
        Materialise all pipelines and optional names.
        Use when Experiment must see all systems together (e.g., significance tests).
        """
        gens = self._normalize_generators(pipeline_generators, "pipeline_generators")

        pipelines: list[Transformer] = []
        names: list[Optional[str]] = []

        def _emit_item_to_lists(item):
            if isinstance(item, tuple) and len(item) == 2:
                p, nm = item
            else:
                p, nm = item, None

            if isinstance(p, Transformer):
                pipelines.append(p)
                names.append(nm if isinstance(nm, str) else None)
            elif isinstance(p, runtime_Sequence) and all(isinstance(pi, Transformer) for pi in p):
                if isinstance(nm, str):
                    pipelines.extend(p)
                    names.extend([nm] * len(p))
                elif isinstance(nm, runtime_Sequence):
                    nm_list = list(nm)
                    if len(nm_list) != len(p):
                        raise ValueError("Length of names does not match number of pipelines.")
                    pipelines.extend(p)
                    names.extend([n if isinstance(n, str) else None for n in nm_list])
                else:
                    pipelines.extend(p)
                    names.extend([None] * len(p))
            else:
                raise ValueError(f"Generator yielded an invalid item: {type(p)}")

        for gen in gens:
            out = gen(context)
            if inspect.isgenerator(out) or isinstance(out, Iterator):
                for item in out:
                    _emit_item_to_lists(item)
            else:
                if isinstance(out, tuple):
                    _pipelines, *_rest = out
                    _names = None if not _rest else _rest[0]
                    _emit_item_to_lists((_pipelines, _names))
                else:
                    _emit_item_to_lists(out)

        if not pipelines:
            raise ValueError("No pipelines generated. Ensure your generators produce valid Transformers.")

        final_names = None if not any(names) else [
            nm if nm is not None else f"pipeline_{i}" for i, nm in enumerate(names)
        ]
        return pipelines, final_names

    # ---------------------------
    # Aggregation utilities
    # ---------------------------
    def compute_overall_mean(
        self,
        results: pd.DataFrame,
        eval_metrics: Sequence[Any] = None,
    ) -> pd.DataFrame:
        measure_cols = [str(m) for m in (eval_metrics or self.__default_measures) if str(m) in results.columns]
        if measure_cols:
            per_ds = (
                results
                .groupby(["dataset", "name"], dropna=False)[measure_cols]
                .mean()
                .reset_index()
            )

            gmean_rows = []
            for name, group in per_ds.groupby("name", dropna=False):
                row = {"dataset": "Overall", "name": name}
                for col in measure_cols:
                    vals = pd.to_numeric(group[col], errors="coerce").dropna().values
                    if np.any(vals <= 0):
                        vals = vals + 1e-12
                    row[col] = geometric_mean(vals)
                gmean_rows.append(row)

            gmean_df = pd.DataFrame(gmean_rows)
            results = pd.concat([results, gmean_df], ignore_index=True)

        return results

    @cache
    def get_measures(self, dataset: str) -> list[Measure]:
        """
        Resolve measures for the given dataset name.
        If the suite maintains a single list, return it; otherwise look up per-dataset.
        """
        if isinstance(self._measures, list):
            return self._measures
        if dataset not in self._measures:
            return self.__default_measures
        return self._measures[dataset]

    @property
    def datasets(self) -> Generator[Tuple[str, pt.datasets.Dataset], None, None]:
        """
        Generator yielding (dataset_name, pt.get_dataset("irds:<id>"))
        """
        if isinstance(self._datasets, list):
            for ds_id in self._datasets:
                yield ds_id, pt.get_dataset(f"irds:{ds_id}")
        elif isinstance(self._datasets, dict):
            for name, ds_id in self._datasets.items():
                yield name, pt.get_dataset(f"irds:{ds_id}")
        else:
            raise ValueError("Suite _datasets must be a list or dict mapping names to dataset IDs.")

    # ---------------------------
    # Internal helpers
    # ---------------------------
    @staticmethod
    def _topics_qrels(ds: pt.datasets.Dataset, query_field: Optional[str]):
        topics = ds.get_topics(query_field, tokenise_query=False)
        qrels = ds.get_qrels()
        return topics, qrels

    @staticmethod
    def _free_cuda():
        import gc
        gc.collect()
        try:
            import torch  # noqa: WPS433 — optional dependency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------------------
    # Main entry point
    # ---------------------------
    def __call__(
        self,
        ranking_generators: Union[Callable[[DatasetContext], Any], Sequence[Callable[[DatasetContext], Any]]],
        eval_metrics: Sequence[Any] = None,
        subset: Optional[str] = None,
        **experiment_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        results: list[pd.DataFrame] = []

        baseline = experiment_kwargs.get("baseline", None)
        coerce_grouped = baseline is not None
        if coerce_grouped:
            logging.warning("Significance tests require pipelines to be grouped; this uses more memory.")

        for corpus_id, corpus_ds, members in self._iter_corpus_groups():
            # If a subset was requested, skip this corpus unless it contains the subset
            if subset and all(name != subset for name, _ in members):
                continue

            # Single shared context per corpus (indexing happens once here)
            context = DatasetContext(corpus_ds)

            if coerce_grouped:
                # Materialise all pipelines ONCE for the corpus
                pipelines, names = self.coerce_pipelines_grouped(context, ranking_generators)

                # Evaluate the same systems across each dataset that shares this corpus
                for ds_name, ds_id in members:
                    if subset and ds_name != subset:
                        continue

                    ds_member = pt.get_dataset(f"irds:{ds_id}")
                    topics, qrels = self._topics_qrels(ds_member, self._query_field)

                    save_dir = experiment_kwargs.pop("save_dir", None)
                    if save_dir is not None:
                        formatted_ds_name = ds_name.replace("/", "-").lower()
                        ds_save_dir = f"{save_dir}/{formatted_ds_name}"
                        experiment_kwargs["save_dir"] = ds_save_dir

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

                # Release materialised pipelines after all member datasets are processed
                try:
                    del pipelines, names
                finally:
                    self._free_cuda()

            else:
                # Stream pipelines one at a time, but reuse each pipeline across ALL member datasets
                for pipeline, name in self.coerce_pipelines_sequential(context, ranking_generators):
                    for ds_name, ds_id in members:
                        if subset and ds_name != subset:
                            continue

                        ds_member = pt.get_dataset(f"irds:{ds_id}")
                        topics, qrels = self._topics_qrels(ds_member, self._query_field)

                        df = pt.Experiment(
                            [pipeline],
                            eval_metrics=eval_metrics or self.get_measures(ds_name),
                            topics=topics,
                            qrels=qrels,
                            names=None if name is None else [name],
                            **experiment_kwargs,
                        )
                        df["dataset"] = ds_name
                        results.append(df)

                    # Dispose of this pipeline (after all member datasets)
                    try:
                        del pipeline
                    finally:
                        self._free_cuda()

            # Release per-corpus context
            del context

        results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        # Aggregate geometric mean only across actual Measure columns
        perquery = experiment_kwargs.get("perquery", False)
        if not perquery and not results_df.empty:
            results_df = self.compute_overall_mean(results_df)

        return results_df