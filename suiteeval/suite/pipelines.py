"""
Normalisation of pipeline-generator output.

A pipeline generator is a callable taking a :class:`~suiteeval.context.DatasetContext`
and producing pipelines. It may return or yield a single transformer, a sequence of
transformers, or either of those paired with a name or a sequence of names. The
functions here flatten every accepted shape into ``(pipeline, name)`` pairs so the
rest of the suite only ever deals with one representation.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator, Sequence as runtime_Sequence
import inspect
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from pyterrier import Transformer

from suiteeval.context import DatasetContext

PipelineGenerator = Callable[[DatasetContext], Any]
PipelineGenerators = Union[PipelineGenerator, Sequence[PipelineGenerator]]
NamedPipeline = Tuple[Transformer, Optional[str]]


def normalize_generators(
    pipeline_generators: PipelineGenerators, what: str
) -> list[PipelineGenerator]:
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
            raise TypeError(f"{what} must be a callable or a sequence of callables.")
        return [pipeline_generators]  # type: ignore[list-item]
    if not all(builtins.callable(f) for f in pipeline_generators):  # type: ignore[arg-type]
        raise TypeError(f"All elements of {what} must be callable.")
    return list(pipeline_generators)  # type: ignore[return-value]


def _split_item(item: Any) -> Tuple[Any, Any]:
    """
    Split one generator output into ``(pipelines, names)``.

    A 2-tuple is read as ``(pipelines, names)`` unless its second element is itself
    a :class:`~pyterrier.Transformer`, in which case the whole item is a sequence of
    pipelines.
    """
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and not isinstance(item[1], Transformer)
    ):
        return item
    return item, None


def iter_named_pipelines(item: Any) -> Iterator[NamedPipeline]:
    """
    Flatten one generator output into ``(pipeline, name)`` pairs.

    Args:
        item: A transformer, a sequence of transformers, or either paired with a
            single name or a sequence of names.

    Yields:
        tuple[Transformer, Optional[str]]: The pipeline and its optional name.

    Raises:
        ValueError: If the item is not a transformer or a sequence of transformers,
            or if a name sequence does not match the number of pipelines.
    """
    pipelines, names = _split_item(item)

    if isinstance(pipelines, Transformer):
        yield pipelines, (names if isinstance(names, str) else None)
        return

    if not (
        isinstance(pipelines, runtime_Sequence)
        and all(isinstance(p, Transformer) for p in pipelines)
    ):
        raise ValueError(f"Generator yielded an invalid item: {type(pipelines)}")

    if isinstance(names, str):
        names = [names] * len(pipelines)
    elif isinstance(names, runtime_Sequence):
        names = list(names)
        if len(names) != len(pipelines):
            raise ValueError("Length of names does not match number of pipelines.")
    else:
        names = [None] * len(pipelines)

    for pipeline, name in zip(pipelines, names):
        yield pipeline, (name if isinstance(name, str) else None)


def _drain(output: Any) -> Iterator[NamedPipeline]:
    """Flatten a lazy generator, reporting a leaked ``StopIteration`` clearly."""
    try:
        for item in output:
            yield from iter_named_pipelines(item)
    except RuntimeError as exc:
        # PEP 479 turns leaked StopIteration into RuntimeError; surface a clear message.
        if "StopIteration" not in str(exc):
            raise
        raise ValueError(
            "Pipeline generator raised StopIteration. "
            "Use `return` to end a generator, or catch StopIteration "
            "from `next()` inside the generator."
        ) from exc


def iter_generator_output(
    context: DatasetContext, pipeline_generators: PipelineGenerators
) -> Iterator[NamedPipeline]:
    """
    Run each generator against ``context`` and flatten its output.

    Args:
        context: The shared context for the corpus being evaluated.
        pipeline_generators: A callable or sequence of callables producing pipelines.

    Yields:
        tuple[Transformer, Optional[str]]: Every pipeline with its optional name.
    """
    for generator in normalize_generators(pipeline_generators, "pipeline_generators"):
        output = generator(context)
        if inspect.isgenerator(output) or isinstance(output, Iterator):
            yield from _drain(output)
        else:
            yield from iter_named_pipelines(output)


def fill_names(names: Sequence[Optional[str]]) -> Optional[list[str]]:
    """
    Replace missing names with positional labels.

    Args:
        names: Names aligned with a list of pipelines; entries may be ``None``.

    Returns:
        Optional[list[str]]: The completed names, or ``None`` if none were given.
    """
    if not any(name is not None for name in names):
        return None
    return [
        name if name is not None else f"pipeline_{i}" for i, name in enumerate(names)
    ]


__all__ = [
    "NamedPipeline",
    "PipelineGenerator",
    "PipelineGenerators",
    "fill_names",
    "iter_generator_output",
    "iter_named_pipelines",
    "normalize_generators",
]
