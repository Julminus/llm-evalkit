"""Agreement statistics for two raters: Cohen's kappa, Spearman's rho, confusion matrices.

Pure Python, no NumPy or SciPy: the inputs are a few hundred scores at
most, and every function here fits on a screen, so each number in a report
can be traced by hand. The tests pin the results against values worked out
by hand and, when they are installed, against scikit-learn and SciPy.

Conventions:

* ``a`` is the reference rater (a person, or the judge being compared
  against) and ``b`` the rater under test. Kappa and rho are symmetric, so the
  order only matters for the confusion matrix: rows are ``a``, columns ``b``.
* Weighted kappa uses the distance between *positions* on the scale given
  by ``categories``, which is always the whole scale (1-5 for judge scores),
  not just the values that happen to occur. scikit-learn's
  ``cohen_kappa_score`` uses the observed labels unless told otherwise, so a
  sample with no 4s would put 3 and 5 next to each other.
* A statistic that is undefined returns ``None`` instead of a number: kappa
  when chance agreement is already perfect (both raters gave everything the
  same single score), rho when either rater gave every item the same score.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Sequence
from typing import Literal, TypeVar

__all__ = [
    "SCALE",
    "Weighting",
    "bootstrap_ci",
    "cohen_kappa",
    "confusion_matrix",
    "pearson",
    "ranks",
    "spearman",
]

SCALE: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The judge's score scale."""

Weighting = Literal["none", "linear", "quadratic"]
_C = TypeVar("_C", bound=Hashable)


def _paired(a: Sequence[object], b: Sequence[object]) -> int:
    if len(a) != len(b):
        raise ValueError(f"the two raters have {len(a)} and {len(b)} ratings; they must be paired")
    return len(a)


def confusion_matrix(a: Sequence[_C], b: Sequence[_C], categories: Sequence[_C]) -> list[list[int]]:
    """``m[i][j]`` = items ``a`` put in ``categories[i]`` and ``b`` in ``categories[j]``."""
    _paired(a, b)
    index = {c: i for i, c in enumerate(categories)}
    if len(index) != len(categories):
        raise ValueError("categories must be distinct")
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b, strict=True):
        if x not in index or y not in index:
            bad = x if x not in index else y
            raise ValueError(f"rating {bad!r} is not one of the categories {list(categories)}")
        m[index[x]][index[y]] += 1
    return m


def cohen_kappa(
    a: Sequence[_C],
    b: Sequence[_C],
    categories: Sequence[_C],
    *,
    weights: Weighting = "none",
) -> float | None:
    """Cohen's kappa, optionally weighted by distance on the scale.

    ``kappa = 1 - sum(w * observed) / sum(w * expected)``, where ``expected``
    is what two raters with these marginals would agree on by chance, and the
    disagreement weight ``w`` between categories ``i`` and ``j`` is 1 for any
    disagreement (``"none"``), ``|i - j|`` (``"linear"``) or ``(i - j)**2``
    (``"quadratic"``). 1 is perfect agreement, 0 is chance level, and below 0
    is worse than chance. ``None`` if there are no ratings, or if chance
    agreement is already perfect.
    """
    n = _paired(a, b)
    if n == 0:
        return None
    m = confusion_matrix(a, b, categories)
    k = len(categories)
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> int:
        if weights == "none":
            return int(i != j)
        if weights == "linear":
            return abs(i - j)
        return (i - j) ** 2

    observed = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return None
    return 1.0 - observed / expected


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; tied values share the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        mean_rank = (start + end) / 2 + 1
        for i in order[start : end + 1]:
            out[i] = mean_rank
        start = end + 1
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's r; ``None`` with fewer than 2 pairs or if either side is constant."""
    n = _paired(x, y)
    if n < 2:
        return None
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sxx = math.fsum(d * d for d in dx)
    syy = math.fsum(d * d for d in dy)
    if sxx == 0 or syy == 0:
        return None
    r = math.fsum(p * q for p, q in zip(dx, dy, strict=True)) / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, r))  # float rounding can land a hair outside [-1, 1]


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rho: Pearson's r on the ranks, with ties given average ranks.

    The tie-corrected form matters here: on a 1-5 scale almost every score is
    tied with others, and the textbook shortcut ``1 - 6 sum(d^2) / n(n^2-1)``
    is only exact without ties.
    """
    _paired(a, b)
    return pearson(ranks(a), ranks(b))


def bootstrap_ci(
    a: Sequence[_C],
    b: Sequence[_C],
    statistic: Callable[[Sequence[_C], Sequence[_C]], float | None],
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for ``statistic(a, b)``, resampling pairs.

    Items are resampled with replacement, keeping each ``(a[i], b[i])`` pair
    together. Resamples where the statistic is undefined are dropped; if more
    than half are, the interval is ``None``, since what is left would describe
    an unusual subset rather than the sample. Seeded, so a report gives the
    same interval every time.
    """
    n = _paired(a, b)
    if n < 2 or resamples < 1:
        return None
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        value = statistic([a[i] for i in picks], [b[i] for i in picks])
        if value is not None:
            values.append(value)
    if len(values) * 2 < resamples:
        return None
    values.sort()
    tail = (1 - level) / 2
    lo = values[max(0, math.floor(tail * len(values)))]
    hi = values[min(len(values) - 1, math.ceil((1 - tail) * len(values)) - 1)]
    return lo, hi
