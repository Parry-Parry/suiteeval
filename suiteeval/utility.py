from typing import Sequence


def geometric_mean(sequence: Sequence[float]) -> float:
    """Compute the geometric mean of a sequence of numbers.

    Args:
        sequence (Sequence[float]): A sequence of numbers.

    Returns:
        float: The geometric mean of the sequence.
    """
    product = 1.0
    n = len(sequence)
    for x in sequence:
        product *= x
    return product ** (1.0 / n) if n > 0 else 0.0


def harmonic_mean(sequence: Sequence[float]) -> float:
    """Compute the harmonic mean of a sequence of numbers.

    Args:
        sequence (Sequence[float]): A sequence of numbers.

    Returns:
        float: The harmonic mean of the sequence.
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    denominator = sum(1.0 / x for x in sequence if x > 0)
    return n / denominator if denominator > 0 else 0.0
