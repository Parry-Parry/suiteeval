from pyterrier_dr import FlexIndex
from suiteeval.index.base import TemporaryIndex
from typing import Any, Iterable, Dict


class TemporaryFlexIndex(TemporaryIndex):
    """
    Temporary FlexIndex context manager.

    Example:
        from pyterrier_dr import RetroMAE
        docs = [{'docno': '1', 'text': 'foo'}, ...]
        with TemporaryFlexIndex(RetroMAE.msmarco_distill(), docs) as flex_index:
            # use flex_index in pipelines
            pipeline = model >> flex_index.np_retriever()
            results = pipeline.search("query text")
    """

    def __init__(
        self,
        model: Any,
        documents: Iterable[Dict[str, Any]],
        sim_fn: Any = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.model = model
        self.documents = documents
        self.sim_fn = sim_fn
        self.verbose = verbose

    def _create_index(
        self, model: Any, documents: Iterable[Dict[str, Any]], path: str
    ) -> FlexIndex:
        # Instantiate FlexIndex
        if self.sim_fn is not None:
            flex = FlexIndex(path, sim_fn=self.sim_fn, verbose=self.verbose)
        else:
            flex = FlexIndex(path, verbose=self.verbose)
        # Build and run indexing pipeline
        pipeline = model >> flex.indexer()
        pipeline.index(documents)
        return flex
