import tempfile
import pandas as pd
import ir_datasets as irds


class DatasetContext:
    dataset: irds.Dataset
    path: str = None

    def __init__(self):
        if self.path is None:
            formatted_dataset = self.dataset.id.replace("/", "-")
            self.path = tempfile.mkdtemp(suffix=f"-{formatted_dataset}")
        self.doc_store = self.dataset.doc_store()

    def text(self, text_field: str = None):
        """
        Returns a DatasetTextLookup instance for retrieving document texts.
        If text_field is None, defaults to the document's default text.
        """
        return DatasetTextLookup(self.dataset, text_field)

    def docs_iter(self):
        for doc in self.dataset.docs:
            yield {"docno": doc.doc_id, "text": getattr(doc, self.text_field, doc.default_text())}


class DatasetTextLookup:
    dataset: irds.Dataset
    text_field: str = None

    def __init__(self):
        self.doc_store = self.dataset.doc_store()

    def get_doc(self, doc_id: str) -> str:
        doc = self.doc_store.get(doc_id)
        if doc is None:
            raise KeyError(f"Document {doc_id} not found in dataset {self.dataset.id}.")
        return getattr(doc, self.text_field, doc.default_text())

    def get_many_docs(self, doc_ids: list) -> dict:
        docs = self.doc_store.get_many(doc_ids)
        if docs is None:
            raise KeyError(f"Documents {doc_ids} not found in dataset {self.dataset.id}.")
        return {doc.doc_id: getattr(doc, self.text_field, doc.default_text()) for doc in docs if doc is not None}

    def __call__(self, results: pd.DataFrame) -> str:
        doc_ids = results["docno"].unique().tolist()
        doc_texts = self.get_many_docs(doc_ids)

        results["text"] = results["docno"].map(doc_texts)
        return results

__all__ = ["DatasetContext"]