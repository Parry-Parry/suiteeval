import tempfile
import pandas as pd
import ir_datasets as irds
from pyterrier.datasets._irds import IRDSTextLoader


class DatasetContext:
    dataset: irds.Dataset
    path: str = None

    def __init__(self):
        if self.path is None:
            formatted_dataset = self.dataset.id.replace("/", "-")
            self.path = tempfile.mkdtemp(suffix=f"-{formatted_dataset}")
        self.doc_store = self.dataset.doc_store()

    def text_loader(self, text_field: str = None):
        """
        Returns a IRDSTextLoader instance for retrieving document texts.
        """
        return IRDSTextLoader(self.dataset, text_field)

    def get_corpus_iter(self):
        for doc in self.dataset.docs:
            yield {
                "docno": doc.doc_id,
                "text": getattr(doc, self.text_field, doc.default_text()),
            }


__all__ = ["DatasetContext"]
