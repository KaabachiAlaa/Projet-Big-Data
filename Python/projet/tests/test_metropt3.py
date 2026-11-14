"""Tests de la référence MetroPT-3 (traitement A, fenêtres de 5 minutes)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.common import compare_results
from src.metropt3_reference import (
    ANALOG_VARS,
    BINARY_VARS,
    compute_features,
    compute_features_pandas,
)

ORIGIN = pd.Timestamp("2020-02-01 00:00:00")


# --------------------------------------------------------------------------- outils
def make_frame(timestamps, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"timestamp": pd.to_datetime(list(timestamps))})
    for v in ANALOG_VARS:
        df[v] = rng.normal(50, 3, len(df))
    for v in BINARY_VARS:
        df[v] = rng.integers(0, 2, len(df))
    return df


def regular(n, start=ORIGIN, step_s=10):
    return start + pd.to_timedelta(np.arange(n) * step_s, unit="s")


def write_raw(df, path):
    """Écrit tel quel (timestamps déjà en texte) avec la colonne d'index sans nom du vrai fichier."""
    d = df.copy()
    d.insert(0, "", np.arange(len(d)) * 10)
    d.to_csv(path, index=False)
    return path


def write_csv(df, path, us=False):
    d = df.copy()
    if us:  # M/D/YYYY H:MM:SS, sans %-m (non portable sous Windows)
        d["timestamp"] = [f"{t.month}/{t.day}/{t.year} {t.hour}:{t.minute:02d}:{t.second:02d}" for t in d["timestamp"]]
    else:
        d["timestamp"] = d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return write_raw(d, path)


def run(df, tmp_path, chunksize=10**6, variables=None, us=False):
    report = {}
    path = write_csv(df, tmp_path / "metropt3.csv", us=us)
    feats = compute_features(path, variables or ANALOG_VARS, chunksize=chunksize, report=report)
    return feats, report


def check(feats, df, variables=None):
    """Mêmes clés, même nombre de lignes, valeurs proches (tolérance J1) après tri par clé."""
    f = feats.copy()
    f["window_start"] = pd.to_datetime(f["window_start"])
    compare_results(f, compute_features_pandas(df, variables or ANALOG_VARS), ["window_start"])


# ------------------------------------------------------------------------ les tests
def test_guide_example_five_minutes(tmp_path):
    df = make_frame(regular(5, step_s=60))
    df["TP2"] = [65.0, 67.0, 66.0, 70.0, 72.0]
    feats, _ = run(df, tmp_path)
    row = feats.iloc[0]
    assert len(feats) == 1
    assert row["n"] == 5 and row["TP2_mean"] == 68.0 and row["TP2_max"] == 72.0
    assert row["TP2_std_pop"] == pytest.approx(np.sqrt(6.8))  # 2,61 ; diviseur n, pas n-1


def test_window_starting_exactly_on_a_boundary(tmp_path):
    ts = ["2020-02-01 10:00:00", "2020-02-01 10:04:59", "2020-02-01 10:05:00"]
    feats, _ = run(make_frame(ts), tmp_path)
    assert list(pd.to_datetime(feats["window_start"])) == [
        pd.Timestamp("2020-02-01 10:00:00"), pd.Timestamp("2020-02-01 10:05:00")]
    assert list(feats["n"]) == [2, 1]  # 10:05:00 ouvre la fenêtre suivante (borne basse incluse)


@pytest.mark.parametrize("chunksize", [1, 2, 7, 17, 64, 1000, 10**6])
def test_chunks_of_different_sizes_give_the_same_result(tmp_path, chunksize):
    rng = np.random.default_rng(5)
    n = 150 if chunksize <= 2 else 800  # peu de lignes quand il y a un bloc par ligne (lecture lente)
    ts = ORIGIN + pd.to_timedelta(np.cumsum(rng.integers(8, 14, n)), unit="s")
    df = make_frame(ts, seed=1)
    feats, _ = run(df, tmp_path, chunksize=chunksize, variables=ANALOG_VARS + BINARY_VARS)
    check(feats, df, ANALOG_VARS + BINARY_VARS)


def test_window_cut_between_two_chunks(tmp_path):
    df = make_frame(regular(120, start=pd.Timestamp("2020-02-01 10:02:30")))  # fenêtres à cheval
    feats, _ = run(df, tmp_path, chunksize=17)
    check(feats, df)
    assert feats["n"].iloc[0] < 30  # la première fenêtre est bien partielle et reconstituée


def test_last_incomplete_window_is_kept_with_its_real_count(tmp_path):
    df = make_frame(regular(65))  # 650 s : deux fenêtres pleines + 5 lignes
    feats, _ = run(df, tmp_path, chunksize=9)
    assert len(feats) == 3 and feats["n"].iloc[-1] == 5 and feats["n"].sum() == 65
    check(feats, df)


def test_periods_without_observations_are_not_filled_or_dropped(tmp_path):
    ts = list(pd.date_range("2020-02-01 10:00:00", periods=10, freq="30s")) + \
         list(pd.date_range("2020-02-01 10:25:00", periods=10, freq="30s"))
    df = make_frame(ts)
    feats, _ = run(df, tmp_path, chunksize=4)
    starts = list(pd.to_datetime(feats["window_start"]))
    assert starts == [pd.Timestamp("2020-02-01 10:00:00"), pd.Timestamp("2020-02-01 10:25:00")]
    assert feats["n"].min() > 0  # aucune ligne vide inventée
    check(feats, df)


def test_duplicates_are_kept_and_counted_not_removed(tmp_path):
    df = make_frame(regular(6))
    df = pd.concat([df, df.iloc[[2]], df.iloc[[2]]]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    feats, rep = run(df, tmp_path, chunksize=3)
    assert feats["n"].sum() == 8  # 6 + 2 doublons : rien supprimé
    assert rep["adjacent_duplicate_timestamps"] == 2
    check(feats, df)


def test_invalid_sensor_values_are_excluded_from_that_variable_only(tmp_path):
    df = make_frame(regular(60)).astype({"TP2": object})
    df.loc[3, "TP2"] = "abc"
    df.loc[5, "TP2"] = np.nan
    feats, rep = run(df, tmp_path, chunksize=11)
    ref_df = df.copy()
    ref_df["TP2"] = pd.to_numeric(ref_df["TP2"], errors="coerce")
    check(feats, ref_df)
    assert rep["invalid_or_missing"]["TP2"] == 2 and rep["invalid_or_missing"]["TP3"] == 0
    assert feats["n"].sum() == 60  # la ligne reste comptée ; seule TP2 ignore la valeur


def test_unreadable_timestamp_rows_are_excluded_and_reported(tmp_path):
    df = make_frame(regular(20))
    raw = df.copy()
    raw["timestamp"] = raw["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    raw.loc[4, "timestamp"] = "not a date"
    path = write_raw(raw, tmp_path / "bad_ts.csv")
    rep = {}
    feats = compute_features(path, ANALOG_VARS, chunksize=6, report=rep)
    assert rep["rows_read"] == 20 and rep["rows_bad_timestamp"] == 1
    assert feats["n"].sum() == 19
    check(feats, df.drop(index=4))


def test_out_of_order_rows_give_the_same_result_and_are_reported(tmp_path):
    df = make_frame(regular(300), seed=2)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    feats, rep = run(shuffled, tmp_path, chunksize=41)
    assert rep["non_monotonic_steps"] > 0
    check(feats, df)


def test_constant_window_has_exactly_zero_std(tmp_path):
    df = make_frame(regular(30))
    df["TP2"] = -0.012
    feats, _ = run(df, tmp_path, chunksize=7)
    assert feats["TP2_std_pop"].iloc[0] <= 1e-12


def test_numerically_stable_with_large_offset_and_small_variation(tmp_path):
    df = make_frame(regular(500), seed=4)
    df["TP3"] = 1e6 + np.random.default_rng(9).normal(0, 1e-3, len(df))
    feats, _ = run(df, tmp_path, chunksize=23)
    check(feats, df)


def test_variable_entirely_invalid_in_a_window_gives_empty_stats_not_zero(tmp_path):
    df = make_frame(regular(60)).astype({"TP3": object})
    df.loc[0:29, "TP3"] = "x"  # toute la première fenêtre
    feats, _ = run(df, tmp_path, chunksize=8)
    first = feats.iloc[0]
    assert first["n"] == 30 and np.isnan(first["TP3_mean"]) and np.isnan(first["TP3_max"])
    ref_df = df.copy()
    ref_df["TP3"] = pd.to_numeric(ref_df["TP3"], errors="coerce")
    check(feats, ref_df)


def test_iso_and_month_first_formats_give_identical_results(tmp_path):
    df = make_frame(regular(100, start=pd.Timestamp("2020-02-01 00:00:00")))
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a, rep_a = run(df, tmp_path / "a", chunksize=13)
    b, rep_b = run(df, tmp_path / "b", chunksize=13, us=True)
    pd.testing.assert_frame_equal(a, b)
    assert rep_a["timestamp_format"] != rep_b["timestamp_format"]
    assert rep_b["first_timestamp"].startswith("2020-02-01")  # lu mois/jour, pas jour/mois


def test_original_data_file_is_never_modified(tmp_path):
    path = write_csv(make_frame(regular(100)), tmp_path / "orig.csv")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    compute_features(path, ANALOG_VARS, chunksize=10, report={})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
