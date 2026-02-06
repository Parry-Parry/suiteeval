import os
import tempfile
from typing import Union, List, Literal, Optional
import pyterrier as pt


class DatasetContext:
    """
    Holds both a PyTerrier Dataset and a filesystem path (for indexes, caches, etc.).
    """

    def __init__(
        self,
        dataset: pt.datasets.Dataset,
        path: Optional[str] = None,
        save_dir: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
    ):
        """
        Args:
            dataset: The pyterrier Dataset instance (must have `_irds_id`).
            path:    Optional filesystem path to use; if omitted, a temp dir
                     will be created for you.
            save_dir: Optional directory where run files are saved per-dataset.
            dataset_names: List of dataset names associated with this corpus.
        """
        self.dataset = dataset
        self.save_dir = save_dir
        self._dataset_names = dataset_names or []
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

    def exists(self, filename: str) -> bool:
        """
        Check if filename exists in save_dir for ALL sub-datasets.

        Returns True only if the file exists for every dataset in this corpus.
        Returns False if save_dir is None or any dataset is missing the file.

        Args:
            filename: The filename to check for (e.g., "BM25.res.gz").

        Returns:
            True if the file exists for all datasets, False otherwise.
        """
        if self.save_dir is None or not self._dataset_names:
            return False

        for ds_name in self._dataset_names:
            formatted = ds_name.replace("/", "-").lower()
            filepath = os.path.join(self.save_dir, formatted, filename)
            if not os.path.exists(filepath):
                return False
        return True


__all__ = ["DatasetContext"]
