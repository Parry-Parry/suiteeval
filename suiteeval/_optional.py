def pyterrier_dr_available() -> bool:
    """Check if the pyterrier_dr package is available."""
    try:
        import pyterrier_dr  # noqa: F401

        return True
    except ImportError:
        return False


def pyterrier_pisa_available() -> bool:
    """Check if the pyterrier_pisa package is available."""
    try:
        import pyterrier_pisa  # noqa: F401

        return True
    except ImportError:
        return False


def pyterrier_available() -> bool:
    """Check if the pyterrier package is available."""
    try:
        import pyterrier  # noqa: F401

        return True
    except ImportError:
        return False


def pyterrier_splade_available() -> bool:
    """Check if the pyterrier_splade package is available."""
    try:
        import pyt_splade  # noqa: F401

        return True
    except ImportError:
        return False
