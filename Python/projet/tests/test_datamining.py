"""Tests du traitement B : variation, moyenne historique, fenêtres manquantes, ordre temporel."""
import numpy as np
import pandas as pd
import pytest

from src.datamining_features import metropt3_datamining, temporal_features, ukdale_datamining

STEP = pd.Timedelta("5min")
T0 = pd.Timestamp("2020-02-01 10:00:00")


def mp_feats(offsets, values, n=None):
    """Résultats du traitement A : fenêtres aux positions `offsets` (en pas de 5 min)."""
    return pd.DataFrame({
        "window_start": [T0 + k * STEP for k in offsets],
        "n": n if n is not None else [30] * len(offsets),
        "A_mean": values,
    })


def row_at(out, offset):
    return out[out["window_start"] == T0 + offset * STEP].iloc[0]


def test_variation_matches_the_guide_example():
    out = metropt3_datamining(mp_feats([0, 1, 2, 3], [65.0, 68.0, 75.0, 70.0]))
    v = out["A_mean_variation"].tolist()
    assert np.isnan(v[0]) and v[1:] == [3.0, 7.0, -5.0]


def test_missing_window_stays_identifiable_and_is_not_filled():
    out = metropt3_datamining(mp_feats([0, 1, 3], [65.0, 68.0, 70.0]))  # 10:10 absente
    assert len(out) == 4  # la grille complète est conservée
    gap = row_at(out, 2)
    assert gap["is_missing_window"] and gap["n"] == 0 and np.isnan(gap["A_mean"])
    assert out["is_missing_window"].tolist() == [False, False, True, False]


def test_variation_is_empty_when_current_or_previous_window_is_missing():
    out = metropt3_datamining(mp_feats([0, 1, 3, 4], [65.0, 68.0, 70.0, 74.0]))
    assert np.isnan(row_at(out, 2)["A_mean_variation"])   # fenêtre absente
    assert np.isnan(row_at(out, 3)["A_mean_variation"])   # précédente absente : pas 70 - 68
    assert row_at(out, 3)["prev_window_missing"]
    assert row_at(out, 4)["A_mean_variation"] == 4.0      # 74 - 70, deux fenêtres présentes


def test_history_is_defined_in_time_not_in_rows():
    """Cinq lignes précédentes != cinq minutes précédentes quand il y a des trous."""
    # 5 min d'historique sur fenêtres d'1 s = 300 fenêtres ; ici on prend 3 fenêtres de 5 min.
    offsets = [0, 1, 2, 5, 6]                       # trou de 2 fenêtres entre 2 et 5
    out = metropt3_datamining(mp_feats(offsets, [10.0, 20.0, 30.0, 100.0, 110.0]), history_minutes=15)
    at5 = row_at(out, 5)                            # historique = positions 2,3,4 -> seule la 2 existe
    assert at5["A_mean_hist_mean"] == 30.0 and at5["hist_n_windows"] == 1 and not at5["hist_complete"]
    at6 = row_at(out, 6)                            # positions 3,4,5 -> seule la 5 existe
    assert at6["A_mean_hist_mean"] == 100.0
    # une version "par lignes" aurait donné la moyenne de 20, 30, 100 pour la position 6
    assert at6["A_mean_hist_mean"] != pytest.approx((20 + 30 + 100) / 3)


def test_history_mean_is_weighted_by_the_number_of_measurements():
    out = metropt3_datamining(mp_feats([0, 1, 2], [10.0, 20.0, 99.0], n=[10, 30, 5]), history_minutes=10)
    assert row_at(out, 2)["A_mean_hist_mean"] == pytest.approx((10 * 10 + 30 * 20) / 40)  # pas 15


def test_history_vs_value_and_flags_at_series_start():
    out = metropt3_datamining(mp_feats([0, 1, 2, 3], [1.0, 2.0, 4.0, 8.0]), history_minutes=10)
    first = out.iloc[0]
    assert np.isnan(first["A_mean_hist_mean"]) and first["hist_n_windows"] == 0 and not first["hist_complete"]
    third = row_at(out, 2)
    assert third["hist_complete"] and third["A_mean_hist_mean"] == 1.5
    assert third["A_mean_vs_hist"] == pytest.approx(4.0 - 1.5)


def test_no_future_leakage():
    base = mp_feats(list(range(10)), list(np.arange(10.0)))
    changed = base.copy()
    changed.loc[7, "A_mean"] = 1e6  # on modifie la fenêtre 7
    a = metropt3_datamining(base, history_minutes=15)
    b = metropt3_datamining(changed, history_minutes=15)
    cols = ["A_mean_variation", "A_mean_hist_mean", "A_mean_vs_hist"]
    pd.testing.assert_frame_equal(a.iloc[:7][cols], b.iloc[:7][cols])  # le passé ne bouge pas
    assert not a.iloc[7:][cols].equals(b.iloc[7:][cols])


def test_row_order_of_the_input_does_not_matter_and_input_is_not_modified():
    f = mp_feats([0, 1, 2, 4, 5], [1.0, 2.0, 4.0, 8.0, 16.0])
    shuffled = f.sample(frac=1.0, random_state=1).reset_index(drop=True)
    snapshot = shuffled.copy()
    pd.testing.assert_frame_equal(metropt3_datamining(f), metropt3_datamining(shuffled))
    pd.testing.assert_frame_equal(shuffled, snapshot)


def test_duplicate_window_keys_are_rejected():
    f = mp_feats([0, 1, 1], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        metropt3_datamining(f)


def test_history_not_multiple_of_the_window_is_rejected():
    with pytest.raises(ValueError):
        metropt3_datamining(mp_feats([0, 1], [1.0, 2.0]), history_minutes=7)


def test_metropt3_default_history_equals_previous_window():
    """Avec des fenêtres de 5 min, 'les 5 minutes précédentes' = la fenêtre précédente."""
    out = metropt3_datamining(mp_feats([0, 1, 2], [65.0, 68.0, 75.0]))
    assert out["A_mean_hist_mean"].tolist()[1:] == [65.0, 68.0]


def test_nan_indicator_in_a_present_window_is_not_counted_as_zero():
    f = mp_feats([0, 1, 2], [10.0, np.nan, 30.0])
    out = metropt3_datamining(f, history_minutes=10)
    assert row_at(out, 2)["A_mean_hist_mean"] == 10.0  # la fenêtre 1 est ignorée, pas comptée comme 0
    assert np.isnan(row_at(out, 2)["A_mean_variation"])


# ------------------------------------------------------------------------- UK-DALE
def uk_feats(indices, i_rms):
    return pd.DataFrame({
        "window_index": indices, "n_samples": 16_000, "is_complete": True,
        "i_rms": i_rms, "u_rms": 230.0, "p_mean": [100.0 * x for x in i_rms],
    })


def test_ukdale_gap_in_window_index_is_made_explicit():
    out = ukdale_datamining(uk_feats([0, 1, 3, 4], [1.0, 2.0, 4.0, 8.0]), history_minutes=3 / 60)
    assert out["window_index"].tolist() == [0, 1, 2, 3, 4]
    assert out["is_missing_window"].tolist() == [False, False, True, False, False]
    assert out.loc[2, "n_samples"] == 0 and not out.loc[2, "is_complete"]
    assert np.isnan(out.loc[3, "i_rms_variation"]) and out.loc[4, "i_rms_variation"] == 4.0
    assert out.loc[4, "i_rms_hist_mean"] == pytest.approx((2.0 + 4.0) / 2)  # positions 1,2,3 -> 1 et 3


def test_ukdale_five_minutes_means_300_windows():
    n = 400
    out = ukdale_datamining(uk_feats(list(range(n)), list(np.arange(n, dtype=float))), history_minutes=5)
    assert out.loc[299, "hist_n_windows"] == 299 and not out.loc[299, "hist_complete"]
    assert out.loc[300, "hist_n_windows"] == 300 and out.loc[300, "hist_complete"]
    assert out.loc[300, "i_rms_hist_mean"] == pytest.approx(np.arange(300).mean())


def test_temporal_features_requires_consistent_positions():
    f = pd.DataFrame({"n": [5, 5], "x": [1.0, 2.0]})
    out = temporal_features(f, pd.Series([0, 1]), ["x"], "n", 1)
    assert out["x_variation"].tolist()[1] == 1.0
