from pyterrier_pisa import PisaIndex
from suiteeval.index.base import TemporaryIndex
from typing import Any, Iterable, Dict


class TemporaryPisaIndex(TemporaryIndex):
    """
    Temporary PISAIndex context manager.

    Example:
        docs = [{'docno': '1', 'text': 'foo'}, ...]
        with TemporaryPisaIndex(docs) as pisa_index:
            # use pisa_index in pipelines
            bm25 = pisa_index.bm25()
            results = bm25.search("query text")
    """

    def __init__(
        self, documents: Iterable[Dict[str, Any]], fields: Any = None, **kwargs
    ):
        super().__init__()
        self.documents = documents
        self.fields = fields
        self.kwargs = kwargs

    def _create_index(self, path: str) -> PisaIndex:
        # model is the PISAIndex class
        if self.fields is not None:
            pisa = PisaIndex(path, fields=self.fields, **self.kwargs)
        else:
            pisa = PisaIndex(path, **self.kwargs)
        # Build and run indexing
        pisa.index(self.documents)
        return pisa

class TemporarySPLADEIndex(TemporaryIndex):
    """
    Temporary PISAIndex context manager.

    Example:
        docs = [{'docno': '1', 'text': 'foo'}, ...]
        with TemporaryPisaIndex(docs) as pisa_index:
            # use pisa_index in pipelines
            bm25 = pisa_index.bm25()
            results = bm25.search("query text")
    """

    def __init__(
        self, model, documents: Iterable[Dict[str, Any]], fields: Any = None, **kwargs
    ):
        super().__init__()
        self.model = model
        self.documents = documents
        self.fields = fields
        self.kwargs = kwargs

    def _create_index(self, path: str) -> PisaIndex:
        # model is the PISAIndex class
        if self.fields is not None:
            pisa = PisaIndex(path, fields=self.fields, stemmer='none' **self.kwargs)
        else:
            pisa = PisaIndex(path, **self.kwargs)
        index_pipeline = self.model.doc_encoder() >> pisa
        # Build and run indexing
        index_pipeline.index(self.documents)
        return self.model.query_encoder() >> pisa.quantized()
