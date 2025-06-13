import tempfile
import ir_datasets as irds


class DatasetContext:
    dataset: irds.Dataset
    path: str = None

    def __init__(self):
        if self.path is None:
            formatted_dataset = self.dataset.id.replace("/", "-")
            self.path = tempfile.mkdtemp(suffix=f"-{formatted_dataset}")

    def doc_iter(self):
        for doc in self.dataset.docs:
            yield {"docno": doc.doc_id, "text": doc.default_text()}
