import tempfile
import shutil
from typing import Any, Iterable


class Temporary:
    """
    Class for a temporary index context manager.
    """

    index_cls = None
    index_kwargs = {}
    _dir = None
    index = None

    def __init__(self, index_cls: Any, **index_kwargs):
        """
        Initialize the temporary index context manager.

        :param index_cls: The class of the index to create.
        :param index_kwargs: Additional keyword arguments for the index class.
        """
        self.index_cls = index_cls
        self.index_kwargs = index_kwargs

    def __enter__(self):
        # Create temporary directory
        self._dir = tempfile.mkdtemp()
        # Build the index
        self.index = self._create_index(self._dir)
        return self.index

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup temporary directory
        self._cleanup(self._dir)

    def _create_index(self, path: str) -> Any:
        """
        Instantiate and build the index in `path` using `model` on `documents`.
        Returns the index object.
        """
        return self.index_cls(path, **self.index_kwargs)

    def _cleanup(self, path: str):
        """
        Remove the temporary directory and its contents.
        """
        shutil.rmtree(path, ignore_errors=True)
