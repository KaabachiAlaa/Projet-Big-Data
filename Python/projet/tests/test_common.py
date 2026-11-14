"""Tests de src/common.py : tolérance J1, comparaison par clé, fusion stable des blocs."""
import numpy as np
import pandas as pd
import pytest

from src.common import ABS_TOL, REL_TOL, compare_results, finalize_moments, merge_moments, within_tolerance


def test_tolerance_rule_is_the_j1_rule():
    assert ABS_TOL == 1e-9 and REL_TOL == 1e-9
    r = 1000.0
    limit = 1e-9 + 1e-9 * abs(r)
    assert within_tolerance(r + 0.99 * limit, r)
    assert not within_tolerance(r + 1.01 * limit, r)
    # l'écart est mesuré par rapport à la référence r, pas à v
    assert within_tolerance(0.0, 5e-10) and not within_tolerance(0.0, 2e-9)


def test_tolerance_treats_two_nan_as_equal_but_not_nan_vs_number():
    assert within_tolerance(np.nan, np.nan)
    assert not within_tolerance(np.nan, 1.0)
    assert not within_tolerance(1.0, np.nan)


def _frame(keys, values):
    return pd.DataFrame({"k": keys, "x": values})


def test_compare_results_sorts_by_key_before_comparing():
    ref = _frame([1, 2, 3], [10.0, 20.0, 30.0])
    res = _frame([3, 1, 2], [30.0, 10.0, 20.0])  # même contenu, autre ordre
    compare_results(res, ref, ["k"])


def test_compare_results_detects_row_count_keys_and_values():
    ref = _frame([1, 2, 3], [10.0, 20.0, 30.0])
    with pytest.raises(AssertionError):
        compare_results(_frame([1, 2], [10.0, 20.0]), ref, ["k"])           # nombre de lignes
    with pytest.raises(AssertionError):
        compare_results(_frame([1, 2, 4], [10.0, 20.0, 30.0]), ref, ["k"])  # clés différentes
    with pytest.raises(AssertionError):
        compare_results(_frame([1, 2, 3], [10.0, 20.0, 30.1]), ref, ["k"])  # valeur hors tolérance
    compare_results(_frame([1, 2, 3], [10.0, 20.0, 30.0 + 1e-12]), ref, ["k"])  # dans la tolérance


def _moments(values_by_window: dict) -> pd.DataFrame:
    rows = {}
    for w, x in values_by_window.items():
        x = np.asarray(x, dtype=float)
        rows[w] = {"n": float(len(x)), "mean": x.mean(), "m2": ((x - x.mean()) ** 2).sum(), "max": x.max()}
    return pd.DataFrame.from_dict(rows, orient="index")


def test_merge_moments_equals_global_computation_for_any_split():
    rng = np.random.default_rng(3)
    x = rng.normal(60, 5, 1000)
    for cut in (1, 7, 333, 999):
        acc = merge_moments(None, _moments({0: x[:cut]}))
        acc = merge_moments(acc, _moments({0: x[cut:]}))
        f = finalize_moments(acc).iloc[0]
        assert f["n"] == len(x)
        assert within_tolerance(f["mean"], x.mean())
        assert within_tolerance(f["std_pop"], x.std(ddof=0))
        assert f["max"] == x.max()


def test_merge_moments_keeps_windows_present_on_one_side_only():
    acc = _moments({0: [1.0, 2.0], 1: [5.0]})
    new = _moments({1: [7.0], 2: [9.0, 9.0, 9.0]})
    out = finalize_moments(merge_moments(acc, new))
    assert list(out.index) == [0, 1, 2]
    assert list(out["n"]) == [2, 2, 3]
    assert out.loc[0, "mean"] == 1.5 and out.loc[2, "mean"] == 9.0
    assert out.loc[1, "mean"] == 6.0 and out.loc[1, "max"] == 7.0


def test_constant_window_has_zero_std_even_when_split():
    """Régression : sumsq/n - mean**2 donnerait ~1e-8, au-delà de la tolérance J1."""
    x = np.full(1000, 0.123456789)
    acc = merge_moments(None, _moments({0: x[:400]}))
    acc = merge_moments(acc, _moments({0: x[400:]}))
    assert finalize_moments(acc).loc[0, "std_pop"] <= 1e-12


def test_std_is_population_std_not_sample_std():
    x = np.array([65.0, 67.0, 66.0, 70.0, 72.0])  # exemple du guide
    f = finalize_moments(merge_moments(None, _moments({0: x}))).iloc[0]
    assert f["mean"] == 68.0 and f["max"] == 72.0
    assert f["std_pop"] == pytest.approx(2.6077, abs=1e-4)
    assert f["std_pop"] != pytest.approx(x.std(ddof=1))
