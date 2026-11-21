from itertools import product

import numpy as np
import pytest

from src.evaluate import best_threshold, expected_cost


def test_threshold_cost_includes_missed_amount_and_false_alerts():
    assert expected_cost([1, 0, 1], [0.9, 0.5, 0.2], [30, 500, 70], 0.5) == 75.0


def test_exact_search_matches_direct_cost_and_keeps_equal_scores_together():
    rng = np.random.default_rng(42)
    for fp_cost, seed in product((0, 2, 5, 10), range(4)):
        labels = rng.integers(0, 2, 12)
        scores = rng.integers(0, 5, 12)
        amounts = rng.integers(0, 20, 12)
        cutoff, minimum, grid, costs = best_threshold(labels, scores, amounts, fp_cost)
        np.testing.assert_array_equal(grid, np.r_[np.inf, np.unique(scores)[::-1]])
        direct_costs = [expected_cost(labels, scores, amounts, t, fp_cost) for t in grid]
        np.testing.assert_array_equal(costs, direct_costs)
        assert cutoff == grid[np.argmin(direct_costs)]
        assert minimum == min(direct_costs)


def test_search_can_choose_no_alerts_and_prefers_fewer_alerts_on_a_tie():
    cutoff, cost, _, _ = best_threshold([0, 1], [0.8, 0.8], [12, 5], fp_cost=5)
    assert cutoff == np.inf
    assert cost == 5.0


def test_search_can_choose_to_alert_on_every_transaction():
    cutoff, cost, _, _ = best_threshold([0, 1], [0.8, 0.1], [12, 50], fp_cost=5)
    assert cutoff == 0.1
    assert cost == 5.0


def test_cost_rejects_amounts_from_a_transformation_with_negative_values():
    with pytest.raises(ValueError, match="Original amounts"):
        expected_cost([1], [0.5], [-1], 0.2)

