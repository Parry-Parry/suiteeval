import tempfile
from typing import Union, List, Literal, Optional
import pyterrier as pt


class DatasetContext:
    """
    Holds both a PyTerrier Dataset and a filesystem path (for indexes, caches, etc.).
    """

    def __init__(self, dataset: pt.datasets.Dataset, path: Optional[str] = None):
        """
        Args:
            dataset: The pyterrier Dataset instance (must have `_irds_id`).
            path:    Optional filesystem path to use; if omitted, a temp dir
                     will be created for you.
        """
        self.dataset = dataset
        if path is None:
            formatted = self.dataset._irds_id.replace("/", "-")
            self.path = tempfile.mkdtemp(suffix=f"-{formatted}")
        else:
            self.path = path

    def text_loader(self, fields: Union[List[str], str, Literal["*"]] = "*"):
        """
        Returns a IRDSTextLoader instance for retrieving document texts.

        Args:
            fields: Fields to load; can be a list of field names, a single
                    field name, or "*" for all fields.
        Returns:
            An IRDSTextLoader instance.
        """
        return self.dataset.text_loader(fields=fields)

    def get_corpus_iter(self, **iter_kwargs):
        """
        Returns an iterator over the corpus documents.

        Args:
            **iter_kwargs: Keyword arguments passed to `get_corpus_iter`.
        """
        return self.dataset.get_corpus_iter(**iter_kwargs)


__all__ = ["DatasetContext"]
