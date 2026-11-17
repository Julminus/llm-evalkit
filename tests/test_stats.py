import math
import random

import pytest

from evalkit.stats import (
    SCALE,
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    pearson,
    ranks,
    spearman,
)


class TestConfusionMatrix:
    def test_rows_are_the_first_rater(self) -> None:
        m = confusion_matrix([1, 1, 2], [1, 2, 2], [1, 2])
        assert m == [[1, 1], [0, 1]]

    def test_unknown_rating_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="not one of the categories"):
            confusion_matrix([1, 6], [1, 2], SCALE)

    def test_unpaired_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="must be paired"):
            confusion_matrix([1, 2], [1], SCALE)

    def test_categories_must_be_distinct(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            confusion_matrix([1], [1], [1, 1])


class TestKappa:
    def test_textbook_unweighted(self) -> None:
        # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.7, pe 0.5.
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        assert cohen_kappa(a, b, ["y", "n"]) == pytest.approx(0.4)

    @pytest.mark.parametrize(
        ("weights", "expected"), [("none", 0.5), ("linear", 2 / 3), ("quadratic", 0.8)]
    )
    def test_worked_by_hand(self, weights: str, expected: float) -> None:
        # a = 1 2 3, b = 1 3 3 on a 1-3 scale. One disagreement (2 vs 3), so the
        # weighted observed disagreement is 1/3 for all three weightings; the
        # chance-expected disagreement is 6/9 (none), 9/9 (linear), 15/9 (quadratic).
        assert cohen_kappa([1, 2, 3], [1, 3, 3], [1, 2, 3], weights=weights) == pytest.approx(  # type: ignore[arg-type]
            expected
        )

    def test_perfect_agreement_is_one(self) -> None:
        a = [1, 2, 3, 4, 5, 3]
        assert cohen_kappa(a, a, SCALE, weights="quadratic") == pytest.approx(1.0)

    def test_undefined_when_chance_agreement_is_perfect(self) -> None:
        assert cohen_kappa([4, 4, 4], [4, 4, 4], SCALE) is None
        assert cohen_kappa([], [], SCALE) is None

    def test_one_rater_constant_is_zero_not_undefined(self) -> None:
        # Agreeing only by always saying 4 earns no credit above chance.
        assert cohen_kappa([4, 4, 4, 4], [4, 5, 3, 4], SCALE) == pytest.approx(0.0)

    def test_worse_than_chance_is_negative(self) -> None:
        assert cohen_kappa([1, 5, 1, 5], [5, 1, 5, 1], SCALE, weights="quadratic") < 0  # type: ignore[operator]

    def test_distance_is_on_the_whole_scale_not_the_observed_labels(self) -> None:
        # No 4s anywhere. On the observed labels (1, 2, 3, 5), a 3-vs-5
        # disagreement would count as one step, like 1-vs-2; on the scale it is two.
        a = [3, 3, 5, 5, 3, 1, 2]
        b = [3, 5, 5, 5, 3, 2, 2]
        full = cohen_kappa(a, b, SCALE, weights="linear")
        squeezed = cohen_kappa(a, b, [1, 2, 3, 5], weights="linear")
        assert full is not None
        assert squeezed is not None
        assert full != pytest.approx(squeezed)


class TestRanksAndCorrelation:
    def test_average_ranks_for_ties(self) -> None:
        assert ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
        assert ranks([]) == []

    def test_spearman_with_ties_by_hand(self) -> None:
        # ranks b = 1 2.5 2.5 4 5: r = 9.5 / sqrt(10 * 9.5)
        assert spearman([1, 2, 3, 4, 5], [1, 2, 2, 4, 5]) == pytest.approx(9.5 / math.sqrt(95))

    def test_monotone_and_reversed(self) -> None:
        assert spearman([1, 2, 3], [10, 20, 90]) == pytest.approx(1.0)
        assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_undefined_cases(self) -> None:
        assert spearman([3, 3, 3], [1, 2, 3]) is None
        assert pearson([1.0], [2.0]) is None


class TestBootstrap:
    def test_interval_contains_the_estimate_and_is_reproducible(self) -> None:
        rng = random.Random(1)
        a = [rng.randint(1, 5) for _ in range(80)]
        b = [min(5, max(1, x + rng.choice([-1, 0, 0, 1]))) for x in a]

        def stat(x, y):  # type: ignore[no-untyped-def]
            return cohen_kappa(x, y, SCALE, weights="quadratic")

        ci = bootstrap_ci(a, b, stat, resamples=500)
        assert ci is not None
        lo, hi = ci
        assert lo <= stat(a, b) <= hi
        assert ci == bootstrap_ci(a, b, stat, resamples=500)
        assert ci != bootstrap_ci(a, b, stat, resamples=500, seed=1)

    def test_none_when_mostly_undefined(self) -> None:
        # Nearly constant: most resamples have one rater constant -> rho undefined.
        a = [1] + [3] * 39
        b = [2] * 40
        assert bootstrap_ci(a, b, spearman, resamples=200) is None

    def test_too_few_items(self) -> None:
        assert bootstrap_ci([1], [1], spearman) is None

    def test_bad_level(self) -> None:
        with pytest.raises(ValueError, match="level"):
            bootstrap_ci([1, 2], [1, 2], spearman, level=1.5)


class TestAgainstReferenceLibraries:
    """Optional cross-check: runs only where scikit-learn / SciPy are installed
    (``uv run --with scikit-learn --with scipy pytest tests/test_stats.py``)."""

    @staticmethod
    def samples() -> list[tuple[list[int], list[int]]]:
        rng = random.Random(7)
        out = []
        for n in (5, 17, 80, 200):
            a = [rng.randint(1, 5) for _ in range(n)]
            b = [min(5, max(1, x + rng.choice([-2, -1, 0, 0, 1]))) for x in a]
            out.append((a, b))
        return out

    def test_kappa_matches_scikit_learn(self) -> None:
        metrics = pytest.importorskip("sklearn.metrics")
        for a, b in self.samples():
            for ours, theirs in (("none", None), ("linear", "linear"), ("quadratic", "quadratic")):
                expected = metrics.cohen_kappa_score(a, b, labels=list(SCALE), weights=theirs)
                assert cohen_kappa(a, b, SCALE, weights=ours) == pytest.approx(expected)  # type: ignore[arg-type]

    def test_spearman_matches_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for a, b in self.samples():
            assert spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)
