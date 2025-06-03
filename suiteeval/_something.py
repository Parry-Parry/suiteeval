import pandas as pd
import pyterrier as pt


class MyAwesomeTransformer(pt.Transformer):
    """A transformer that does something awesome."""

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Do something awesome."""
        return inp
