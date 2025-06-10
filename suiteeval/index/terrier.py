from suiteeval.index.base import TemporaryIndex
from typing import Any, Iterable, Dict
from pyterrier.terrier import IterDictIndexer, TerrierIndex
from pyterrier import IndexFactory


class TemporaryTerrierIndex(TemporaryIndex):
    """
    Temporary TerrierIndex context manager.

    Example:
        docs = [{'docno': '1', 'text': 'foo'}, ...]
        with TemporaryTerrierIndex(docs, indexer_kwargs={'porter2': True}) as terrier_index:
            # use terrier_index in pipelines
            bm25 = pt.BatchRetrieve(terrier_index, wmodel='BM25')
            results = bm25.search("query text")
    """
    def __init__(self, documents: Iterable[Dict[str, Any]], indexer_kwargs: Dict[str, Any] = None, factory_kwargs: Dict[str, Any] = None):
        # model is fixed as TerrierIndex class
        super().__init__()
        self.documents = documents
        self.factory_kwargs = factory_kwargs or {}
        self.indexer_kwargs = indexer_kwargs or {}

    def _create_index(self, documents: Iterable[Dict[str, Any]], path: str) -> TerrierIndex:
        # Instantiate TerrierIndex at path with any provided kwargs
        terrier = IterDictIndexer(path, **self.indexer_kwargs)
        terrier.index(documents)
        return IndexFactory.of(path, **self.factory_kwargs)
