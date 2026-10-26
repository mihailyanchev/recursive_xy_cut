"""
Per-page reading order metrics.

Each metric takes a predicted order as a list of ground-truth positions: ``[1, 0, 2]``
means the element that should be read second was predicted first, and so on. The
ground truth order is therefore always ``[0, 1, ..., n - 1]``.

Kendall's tau and Spearman's rho are undefined for a single element; they are taken to
be 1.0 there, since a single element can only be in the right order.
"""

from itertools import combinations


def exact_match(predicted: list[int]) -> float:
    """Return 1.0 if every element is in the correct position, otherwise 0.0."""
    return float(predicted == list(range(len(predicted))))


def position_accuracy(predicted: list[int]) -> float:
    """Return the fraction of elements predicted at their exact ground-truth position."""
    if not predicted:
        return 1.0
    return sum(truth == position for position, truth in enumerate(predicted)) / len(predicted)


def kendall_tau(predicted: list[int]) -> float:
    """
    Return Kendall's tau between the predicted and the ground-truth order.

    Counts pairs of elements in the same relative order in both (concordant) minus pairs
    in opposite order (discordant), divided by the number of pairs: 1.0 for the correct
    order, -1.0 for the reversed one.
    """
    n = len(predicted)
    if n < 2:
        return 1.0
    concordant_minus_discordant = sum(
        1 if first < second else -1 for first, second in combinations(predicted, 2)
    )
    return concordant_minus_discordant / (n * (n - 1) / 2)


def spearman_rho(predicted: list[int]) -> float:
    """
    Return Spearman's rho between the predicted and the ground-truth order.

    Based on the squared distances between each element's predicted and true position:
    1.0 for the correct order, -1.0 for the reversed one.
    """
    n = len(predicted)
    if n < 2:
        return 1.0
    squared_distances = sum((truth - position) ** 2 for position, truth in enumerate(predicted))
    return 1 - 6 * squared_distances / (n * (n**2 - 1))


METRICS = {
    "Exact match": exact_match,
    "Kendall's τ": kendall_tau,
    "Spearman's ρ": spearman_rho,  # noqa: RUF001
    "Position accuracy": position_accuracy,
}
