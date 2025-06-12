import tempfile
import shutil
from abc import ABC, abstractmethod
from typing import Any
from pyterrier import Transformer


class TemporaryIndex(ABC):
    """
    Abstract base class for a temporary index context manager.
    Subclasses must implement _create_index and _cleanup.
    """

    _dir = None
    index = None

    def __enter__(self):
        # Create temporary directory
        self._dir = tempfile.mkdtemp()
        # Build the index
        self.index = self._create_index(self.model, self.documents, self._dir)
        return self.index

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup index and temporary directory
        try:
            if hasattr(self.index, "close"):
                # Some index implementations may require explicit close
                self.index.close()
        finally:
            self._cleanup(self._dir)

    @abstractmethod
    def _create_index(self, path: str) -> Any:
        """
        Instantiate and build the index in `path` using `model` on `documents`.
        Returns the index object.
        """
        pass

    def _cleanup(self, path: str):
        """
        Remove the temporary directory and its contents.
        """
        shutil.rmtree(path, ignore_errors=True)
