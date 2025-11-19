from __future__ import annotations

from abc import ABCMeta, ABC
import builtins
from collections.abc import Sequence as runtime_Sequence, Iterator
import inspect
import os
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
    Metaclass for :class:`Suite`.

    Responsibilities:
    - Maintain a registry of suite classes by name.
    - Enforce a singleton instance per suite class (i.e., one instance per subclass).
    - Provide a :meth:`register` helper to dynamically create and register suites.
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
            suite_name: Name to assign to the dynamically created suite subclass.
            datasets: IRDS dataset identifiers (e.g., ``"msmarco-passage/trec-dl-2019"``).
            names: Optional display names corresponding one-to-one with ``datasets``.
                Defaults to ``datasets`` when omitted.
            metadata: Optional metadata. Accepted forms:
                * ``None`` → per-dataset empty dicts
                * ``list[dict]`` → each entry applies to the corresponding dataset in ``names``/``datasets``
                * ``dict[str, dict]`` → explicit mapping from dataset name/ID to metadata dict
                * ``dict[str, Any]`` where values are not dicts → treated as flat metadata applied to all
            query_field: Optional topic field name to use when fetching topics (e.g., ``"title"``).

        Returns:
            Suite: The singleton instance of the dynamically created suite class.

        Raises:
            ValueError: If ``metadata`` has an unsupported shape or length.
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
            "_datasets": dataset_map,  # display-name -> dataset_id
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
    Abstract base class for a set of related evaluations across one or more datasets.

    Subclasses (or classes created via :meth:`SuiteMeta.register`) must populate:

    Attributes:
        _datasets: Either a ``dict[str, str]`` mapping display name → IRDS dataset ID,
            or a ``list[str]`` of IRDS dataset IDs.
        _dataset_ids: Normalized mapping of display name → IRDS dataset ID
            (filled in by registration helpers).
        _metadata: Optional per-dataset or global metadata.
        _measures: A list of :class:`ir_measures.Measure` or a mapping from dataset name
            to such a list. When not provided, defaults are derived from metadata or
            IRDS documentation; ultimately falling back to ``[nDCG@10]``.
        _query_field: Optional topic field name to use when fetching topics.

    Notes:
        Instances are singletons per subclass (enforced by :class:`SuiteMeta`).
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
        assert self._datasets, (
            "Suite must have at least one dataset defined in _datasets"
        )

        if not isinstance(self._datasets, (dict, list)):
            raise AssertionError(
                "Suite _datasets must be a dict[name->id] or a list[dataset_id]"
            )
        """
        TODO: Allow validation of non-string datasets
        if isinstance(self._datasets, dict):
            if not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in self._datasets.items()
            ):
                raise AssertionError(
                    "Suite _datasets must map string names to string dataset IDs"
                )
        else:
            if not all(isinstance(ds, str) for ds in self._datasets):
                raise AssertionError(
                    "Suite _datasets list must contain dataset IDs (str)"
                )
        """

        assert self._measures is not None, (
            "Suite must have measures defined in _measures"
        )

    # ---------------------------
    # Corpus grouping
    # ---------------------------
    @staticmethod
    def _get_irds_id(ds_id_or_obj: Any) -> str:
        """
        Extract the IRDS ID from either a string ID or a dataset object.

        Args:
            ds_id_or_obj: Either a string IRDS ID or an object with `_irds_id` attribute.

        Returns:
            str: The IRDS ID.
        """
        if isinstance(ds_id_or_obj, str):
            return ds_id_or_obj
        return ds_id_or_obj._irds_id

    @staticmethod
    def _get_dataset_object(ds_id_or_obj: Any) -> pt.datasets.Dataset:
        """
        Get a PyTerrier Dataset object from either a string ID or a dataset object.

        Args:
            ds_id_or_obj: Either a string IRDS ID or a dataset object.

        Returns:
            pt.datasets.Dataset: The dataset object.
        """
        if isinstance(ds_id_or_obj, str):
            return pt.get_dataset(f"irds:{ds_id_or_obj}")
        return ds_id_or_obj

    def _iter_corpus_groups(self):
        """
        Yield groups of datasets that share the same underlying corpus, determined by
        ir_datasets.docs_parent_id(dataset_id).

        Yields:
            (corpus_id: str,
             corpus_ds: pt.datasets.Dataset,
             members: list[tuple[str, Any]])  # [(display_name, dataset_id_or_obj), ...]
        """
        # normalise to a list of (name, ds_id_or_obj)
        if isinstance(self._datasets, dict):
            items = list(self._datasets.items())
        else:
            items = [(ds_id, ds_id) for ds_id in self._datasets]

        # group by docs-parent (corpus) id
        groups: dict[str, dict] = {}
        for name, ds_id_or_obj in items:
            # Get the IRDS ID and determine corpus parent
            irds_id = self._get_irds_id(ds_id_or_obj)
            try:
                corpus_id = irds.docs_parent_id(irds_id) or irds_id
            except Exception:
                corpus_id = irds_id

            if corpus_id not in groups:
                # Load the corpus dataset (handles both string IDs and objects)
                corpus_ds = self._get_dataset_object(corpus_id if isinstance(corpus_id, str) else irds_id)
                groups[corpus_id] = {
                    "corpus_ds": corpus_ds,
                    "members": [],
                }

            groups[corpus_id]["members"].append((name, ds_id_or_obj))

        # deterministic iteration order (insertion order is fine here)
        for corpus_id, g in groups.items():
            yield corpus_id, g["corpus_ds"], g["members"]

    # ---------------------------
    # Measures
    # ---------------------------
    @staticmethod
    def parse_measures(measures: list[Union[str, Measure]]) -> list[Measure]:
        """
        Convert a list of measure strings or :class:`ir_measures.Measure` objects
        into a flat ``list[Measure]``.

        Args:
            measures: A sequence containing measure strings (e.g., ``"nDCG@10"``)
                and/or :class:`ir_measures.Measure` instances.

        Returns:
            list[Measure]: Parsed measure objects.

        Raises:
            ValueError: If a string entry cannot be parsed by either
                :func:`ir_measures.parse_measure` or :func:`ir_measures.parse_trec_measure`,
                or if an entry has an invalid type.
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
        Populate ``self._measures`` by aggregating available sources in priority order:

        1. Global ``metadata['official_measures']`` if present.
        2. Per-dataset ``metadata[name]['official_measures']`` if present.
        3. IRDS documentation ``official_measures`` for each dataset (when available).

        If no measures are discovered, default to ``[nDCG@10]``.

        Args:
            metadata: The suite metadata dictionary as configured at construction time.

        Returns:
            None
        """

        if self._measures is not None:
            return

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
            names_iter = (
                self._datasets if isinstance(self._datasets, dict) else self._datasets
            )
            for name in names_iter:
                md = metadata.get(name, {})
                if isinstance(md, dict):
                    _add_many(md.get("official_measures"))

        # (3) ir_datasets documentation
        for name, ds_id in (
            self._dataset_ids.items() if isinstance(self._dataset_ids, dict) else []
        ):
            try:
                ds = irds.load(ds_id)
                docs = getattr(ds, "documentation", lambda: None)()
                if isinstance(docs, dict):
                    _add_many(docs.get("official_measures"))
            except Exception as e:
                logging.warning(
                    f"Failed to load measures from documentation for '{name}' ({ds_id}): {e}"
                )

        if not measures_accum:
            logging.warning("No measures discovered; defaulting to [nDCG@10].")
            measures_accum = [nDCG @ 10]

        self._measures = measures_accum

    @staticmethod
    def _normalize_generators(
        pipeline_generators: Union[Callable[[DatasetContext], Any], runtime_Sequence],
        what: str,
    ) -> list[Callable[[DatasetContext], Any]]:
        """
        Normalize a callable or a sequence of callables to a list of callables.

        Args:
            pipeline_generators: Either a single callable taking ``DatasetContext`` and
                yielding pipelines, or a sequence of such callables.
            what: Human-readable label used in error messages.

        Returns:
            list[Callable[[DatasetContext], Any]]: The normalized list.

        Raises:
            TypeError: If the input is neither callable nor a sequence of callables.
        """
        if not isinstance(pipeline_generators, runtime_Sequence) or isinstance(
            pipeline_generators, (str, bytes)
        ):
            if not builtins.callable(pipeline_generators):
                raise TypeError(
                    f"{what} must be a callable or a sequence of callables."
                )
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
        Yield pipelines lazily, one at a time, without materializing the full set.

        Use this when you want to minimize memory/VRAM footprint and you do not require
        joint analysis across all systems at once (e.g., significance testing).

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables that produce either:
                * a single :class:`pyterrier.Transformer`,
                * a sequence of transformers,
                * a tuple ``(pipelines, name_or_names)`` where names may be a single label
                applied to all pipelines or a sequence aligned with ``pipelines``.

        Yields:
            tuple[Transformer, Optional[str]]: The pipeline and an optional display name.

        Raises:
            ValueError: If a generator yields an invalid structure.
        """
        gens = self._normalize_generators(pipeline_generators, "pipeline_generators")

        def _yield_item(item):
            if isinstance(item, tuple) and len(item) == 2:
                p, nm = item
            else:
                p, nm = item, None

            if isinstance(p, Transformer):
                yield p, (nm if isinstance(nm, str) else None)
            elif isinstance(p, runtime_Sequence) and all(
                isinstance(pi, Transformer) for pi in p
            ):
                if isinstance(nm, str):
                    for pi in p:
                        yield pi, nm
                elif isinstance(nm, runtime_Sequence):
                    nm_list = list(nm)
                    if len(nm_list) != len(p):
                        raise ValueError(
                            "Length of names does not match number of pipelines."
                        )
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
        Materialize all pipelines (and optional names) into lists.

        Use this when downstream evaluation requires access to the full set of systems
        simultaneously (e.g., significance tests).

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables following the same
                conventions as in :meth:`coerce_pipelines_sequential`.

        Returns:
            tuple[list[Transformer], Optional[list[str]]]:
                A list of pipelines and, if provided, a list of corresponding names.
                If no names were supplied, returns ``None`` for the second element.

        Raises:
            ValueError: If the generators produce no pipelines or an invalid structure.
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
            elif isinstance(p, runtime_Sequence) and all(
                isinstance(pi, Transformer) for pi in p
            ):
                if isinstance(nm, str):
                    pipelines.extend(p)
                    names.extend([nm] * len(p))
                elif isinstance(nm, runtime_Sequence):
                    nm_list = list(nm)
                    if len(nm_list) != len(p):
                        raise ValueError(
                            "Length of names does not match number of pipelines."
                        )
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
            raise ValueError(
                "No pipelines generated. Ensure your generators produce valid Transformers."
            )

        final_names = (
            None
            if not any(names)
            else [
                nm if nm is not None else f"pipeline_{i}" for i, nm in enumerate(names)
            ]
        )
        return pipelines, final_names

    def compute_overall_mean(
        self,
        results: pd.DataFrame,
        eval_metrics: Sequence[Any] = None,
    ) -> pd.DataFrame:
        """
        Append overall (geometric mean) rows across datasets for each system name.

        This first aggregates per-dataset means over repeated runs, then computes the
        geometric mean across datasets for each metric and appends rows with
        ``dataset == "Overall"``.

        Args:
            results: DataFrame with at least ``["dataset", "name"]`` and metric columns.
            eval_metrics: Optional sequence of metrics to consider; defaults to
                ``self.__default_measures`` when not provided.

        Returns:
            pandas.DataFrame: The input results with additional ``Overall`` rows appended.
        """
        measure_cols = [
            str(m)
            for m in (eval_metrics or self.__default_measures)
            if str(m) in results.columns
        ]
        if measure_cols:
            per_ds = (
                results.groupby(["dataset", "name"], dropna=False)[measure_cols]
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
        Resolve the measures applicable to a given dataset name.

        Args:
            dataset: Dataset display name as used in this suite.

        Returns:
            list[Measure]: The list configured for this dataset (or the suite-wide
                list if a single list is maintained). Falls back to defaults when the
                dataset is unknown.
        """
        if isinstance(self._measures, list):
            return self._measures
        if dataset not in self._measures:
            return self.__default_measures
        return self._measures[dataset]

    @property
    def datasets(self) -> Generator[Tuple[str, pt.datasets.Dataset], None, None]:
        """
        Iterate over declared datasets yielding display name and PyTerrier dataset.

        Yields:
            tuple[str, pyterrier.datasets.Dataset]: Pairs of (name, dataset object).

        Raises:
            ValueError: If ``_datasets`` has an invalid type.
        """
        if isinstance(self._datasets, list):
            for ds_id_or_obj in self._datasets:
                dataset = self._get_dataset_object(ds_id_or_obj)
                yield ds_id_or_obj, dataset
        elif isinstance(self._datasets, dict):
            for name, ds_id_or_obj in self._datasets.items():
                dataset = self._get_dataset_object(ds_id_or_obj)
                yield name, dataset
        else:
            raise ValueError(
                "Suite _datasets must be a list or dict mapping names to dataset IDs."
            )

    @staticmethod
    def _topics_qrels(ds: pt.datasets.Dataset, query_field: Optional[str]):
        """
        Fetch topics and qrels for a dataset.

        Args:
            ds: A :class:`pyterrier.datasets.Dataset` instance.
            query_field: Optional topic field name (e.g., ``"title"``).

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: ``(topics, qrels)``.
        """
        topics = ds.get_topics(query_field)
        qrels = ds.get_qrels()
        return topics, qrels

    @staticmethod
    def _free_cuda():
        """
        Best-effort memory cleanup helper.

        Calls ``gc.collect()`` and, if ``torch.cuda.is_available()``, empties the CUDA cache.
        Silently ignores any exceptions (CUDA and torch are optional).
        """
        import gc

        gc.collect()
        try:
            import torch  # noqa: WPS433 — optional dependency

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __call__(
        self,
        ranking_generators: Union[
            Callable[[DatasetContext], Any], Sequence[Callable[[DatasetContext], Any]]
        ],
        eval_metrics: Sequence[Any] = None,
        subset: Optional[str] = None,
        **experiment_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Run the experiment(s) for each dataset in the suite and return a results table.

        If a ``baseline`` is provided in ``experiment_kwargs``, all pipelines are
        materialized together (grouped mode) to enable tests that require joint access
        (e.g., significance). Otherwise, pipelines are streamed one-by-one to reduce
        memory usage (sequential mode).

        Args:
            ranking_generators: Callable or sequence of callables producing pipelines
                per :class:`DatasetContext` (same conventions as in
                :meth:`coerce_pipelines_sequential`).
            eval_metrics: Optional explicit metrics to evaluate; defaults to the suite’s
                configuration for each dataset.
            subset: Optional dataset display name to restrict evaluation to a single member.
            **experiment_kwargs: Additional keyword arguments forwarded to
                :func:`pyterrier.Experiment`. If ``save_dir`` is provided, it is
                suffixed per dataset.

        Returns:
            pandas.DataFrame: The concatenated experiment results. When ``perquery`` is
                not set, an additional ``Overall`` row is appended per system with
                geometric-mean aggregation across datasets.

        Notes:
            This method reuses a single index per corpus group and cleans up GPU memory
            between pipeline evaluations.
        """
        results: list[pd.DataFrame] = []

        baseline = experiment_kwargs.get("baseline", None)
        coerce_grouped = baseline is not None
        if coerce_grouped:
            logging.warning(
                "Significance tests require pipelines to be grouped; this uses more memory."
            )

        for corpus_id, corpus_ds, members in self._iter_corpus_groups():
            # If a subset was requested, skip this corpus unless it contains the subset
            if subset and all(name != subset for name, _ in members):
                continue

            # Single shared context per corpus (indexing happens once here)
            context = DatasetContext(corpus_ds)

            if coerce_grouped:
                # Materialise all pipelines ONCE for the corpus
                pipelines, names = self.coerce_pipelines_grouped(
                    context, ranking_generators
                )

                # Evaluate the same systems across each dataset that shares this corpus
                for ds_name, ds_id_or_obj in members:
                    if subset and ds_name != subset:
                        continue

                    ds_member = self._get_dataset_object(ds_id_or_obj)
                    topics, qrels = self._topics_qrels(ds_member, self._query_field)

                    save_dir = experiment_kwargs.pop("save_dir", None)
                    if save_dir is not None:
                        if not isinstance(ds_name, str):
                            ds_name = self._get_irds_id(ds_name)
                        formatted_ds_name = ds_name.replace("/", "-").lower()
                        ds_save_dir = f"{save_dir}/{formatted_ds_name}"
                        experiment_kwargs["save_dir"] = ds_save_dir
                        os.makedirs(ds_save_dir, exist_ok=True)

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
                for pipeline, name in self.coerce_pipelines_sequential(
                    context, ranking_generators
                ):
                    for ds_name, ds_id_or_obj in members:
                        if subset and ds_name != subset:
                            continue

                        ds_member = self._get_dataset_object(ds_id_or_obj)
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

        results_df = (
            pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        )

        # Aggregate geometric mean only across actual Measure columns
        perquery = experiment_kwargs.get("perquery", False)
        if not perquery and not results_df.empty:
            results_df = self.compute_overall_mean(results_df)

        return results_df
