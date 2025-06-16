import pytest
import pandas as pd
from suiteeval.context import DatasetContext

class FakeDoc:
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.default_text = lambda: text

class FakeDataset:
    id = "fake/ds"
    def __init__(self):
        self.docs = [FakeDoc("d1", "hello"), FakeDoc("d2", "world")]
    def doc_store(self):
        class Store:
            def get(self, doc_id):
                for doc in FakeDataset().docs:
                    if doc.doc_id == doc_id:
                        return doc
                return None
            def get_many(self, doc_ids):
                return [self.get(doc_id) for doc_id in doc_ids]
        return Store()

def test_dataset_text_lookup_adds_text_column():
    ds = FakeDataset()
    # bypass __init__ to set dataset
    context = object.__new__(DatasetContext)
    context.dataset = ds
    context.text_field = None
    lookup = context.text_loader()

    df = pd.DataFrame({"docno": ["d1", "d2"]})
    out = lookup(df.copy())
    assert "text" in out.columns
    assert set(out["text"]) == {"hello", "world"}
