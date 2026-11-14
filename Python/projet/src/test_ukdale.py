"""Tests de la référence UK-DALE (traitement A, fenêtres de 16 000 échantillons)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.common import compare_results
from src.ukdale_reference import (
    accumulate_blocks,
    compute_features,
    compute_features_numpy,
    finalize,
    read_calibration,
)

W = 16_000
CAL = {"volts_per_adc_step": 1.90101491444e-07, "amps_per_adc_step": 4.9224284384e-08}
COLS = ["window_index", "n_samples", "i_rms", "u_rms", "p_mean"]


# --------------------------------------------------------------------------- outils
def random_raw(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-(2**30), 2**30, size=(n, 2), dtype=np.int64).astype(np.int32)


def sine_raw(n, u_rms=230.0, i_rms=5.0, cos_phi=0.8):
    t = np.arange(n) / 16_000
    u = u_rms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
    i = i_rms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t - np.arccos(cos_phi))
    raw = np.stack([np.round(u / CAL["volts_per_adc_step"]), np.round(i / CAL["amps_per_adc_step"])], axis=1)
    return (raw.astype(np.int64).astype(np.int32)) & ~np.int32(0xFF)  # 24 bits exacts, comme dans un FLAC


def split(raw, sizes):
    out, s = [], 0
    for k in sizes:
        out.append(raw[s:s + k]); s += k
    assert s == len(raw)
    return out


def feats_from_blocks(blocks, window_size=W):
    return finalize(accumulate_blocks(blocks, CAL, window_size), window_size)


def check(feats, raw, window_size=W):
    compare_results(feats[COLS], compute_features_numpy(raw, CAL, window_size)[COLS], ["window_index"])


# ------------------------------------------------------------------------ les tests
def test_window_ending_exactly_on_a_boundary_leaves_no_phantom_window():
    raw = random_raw(2 * W)
    f = feats_from_blocks([raw])
    assert list(f["n_samples"]) == [W, W] and f["is_complete"].all()
    check(f, raw)


def test_window_cut_between_two_blocks_is_rebuilt_from_partial_sums():
    raw = random_raw(3 * W)
    f = feats_from_blocks(split(raw, [10_000, 10_000, 28_000]))  # coupe les fenêtres 0 et 1
    check(f, raw)


def test_last_incomplete_window_is_kept_with_its_real_sample_count():
    raw = random_raw(2 * W + 8_000)
    f = feats_from_blocks(split(raw, [20_000, 20_000]))
    assert list(f["n_samples"]) == [W, W, 8_000]
    assert list(f["is_complete"]) == [True, True, False]
    check(f, raw)  # moyennes calculées sur 8 000 échantillons, pas sur 16 000


@pytest.mark.parametrize("seed", range(5))
def test_random_block_sizes_give_the_same_result(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(W // 2, 4 * W))
    raw = random_raw(n, seed=seed)
    cuts = np.sort(rng.choice(np.arange(1, n), size=int(rng.integers(1, 12)), replace=False))
    sizes = np.diff(np.concatenate([[0], cuts, [n]])).tolist()
    check(feats_from_blocks(split(raw, sizes)), raw)


def test_single_sample_blocks_are_supported():
    raw = random_raw(300)
    check(feats_from_blocks(split(raw, [1] * 300), window_size=100), raw, window_size=100)


def test_known_signal_gives_expected_physical_values():
    f = feats_from_blocks([sine_raw(3 * W)])
    assert f["u_rms"].tolist() == pytest.approx([230.0] * 3, rel=1e-6)
    assert f["i_rms"].tolist() == pytest.approx([5.0] * 3, rel=1e-6)
    assert f["p_mean"].tolist() == pytest.approx([920.0] * 3, rel=1e-6)  # U x I x cos(phi)


def test_calibration_is_applied_before_the_calculation():
    raw = sine_raw(W)
    base = feats_from_blocks([raw]).iloc[0]
    doubled = finalize(accumulate_blocks([raw], {"volts_per_adc_step": 2 * CAL["volts_per_adc_step"],
                                                 "amps_per_adc_step": 3 * CAL["amps_per_adc_step"]}), W).iloc[0]
    assert doubled["u_rms"] == pytest.approx(2 * base["u_rms"])
    assert doubled["i_rms"] == pytest.approx(3 * base["i_rms"])
    assert doubled["p_mean"] == pytest.approx(6 * base["p_mean"])  # P = U x I : facteurs multipliés


def test_channels_are_not_interchangeable():
    raw = sine_raw(W)
    swapped = feats_from_blocks([raw[:, ::-1].copy()]).iloc[0]
    normal = feats_from_blocks([raw]).iloc[0]
    assert normal["u_rms"] == pytest.approx(230.0, rel=1e-6)
    assert swapped["u_rms"] < 150.0  # le "courant" lu comme tension : hors plage secteur plausible


def test_read_calibration_parses_and_rejects_incomplete_files(tmp_path):
    p = tmp_path / "calibration_house_1.cfg"
    p.write_text("[Calibration]\nvolts_per_adc_step = 1.90101491444e-07\n"
                 "amps_per_adc_step = 4.9224284384e-08\nphase_difference = -2.92555759986\n")
    cal = read_calibration(p)
    assert cal["volts_per_adc_step"] == 1.90101491444e-07 and cal["phase_difference"] == -2.92555759986
    p.write_text("volts_per_adc_step = 1.9e-07\n")
    with pytest.raises(ValueError):
        read_calibration(p)


# ------------------------------------------------------------- lecture réelle d'un FLAC
sf = pytest.importorskip("soundfile")


def write_flac(path, raw, samplerate=16_000):
    sf.write(str(path), raw, samplerate, subtype="PCM_24", format="FLAC")
    return path


@pytest.mark.parametrize("blocksize", [777, 12_345, W, 10**6])
def test_flac_read_by_blocks_matches_direct_computation(tmp_path, blocksize):
    raw = sine_raw(3 * W + 1_280)  # 1 280 = 4 périodes de 50 Hz : RMS global exact malgré la fenêtre partielle
    flac = write_flac(tmp_path / "vi-1363392000_236972.flac", raw)
    rep = {}
    f = compute_features(flac, CAL, blocksize=blocksize, report=rep)
    check(f, raw)
    assert rep["total_samples"] == len(raw) and rep["last_window_samples"] == 1_280
    assert rep["mapping_plausible"] is True and rep["global_u_rms_v"] == pytest.approx(230.0, rel=1e-4)


def test_flac_excerpt_with_max_seconds_keeps_the_partial_last_window(tmp_path):
    flac = write_flac(tmp_path / "vi-1363392000_1.flac", sine_raw(4 * W))
    f = compute_features(flac, CAL, blocksize=5_000, max_seconds=2.5)
    assert list(f["n_samples"]) == [W, W, 8_000]


def test_flac_start_time_comes_from_the_file_name(tmp_path):
    flac = write_flac(tmp_path / "vi-1363392000_236972.flac", sine_raw(2 * W))
    f = compute_features(flac, CAL)
    assert list(f["window_start_utc"]) == ["2013-03-16 00:00:00", "2013-03-16 00:00:01"]


def test_wrong_sample_rate_or_channel_count_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        compute_features(write_flac(tmp_path / "slow.flac", sine_raw(W), samplerate=8_000), CAL)
    mono = tmp_path / "mono.flac"
    sf.write(str(mono), sine_raw(W)[:, 0], 16_000, subtype="PCM_24", format="FLAC")
    with pytest.raises(ValueError):
        compute_features(mono, CAL)


def test_implausible_voltage_is_flagged_in_the_report(tmp_path):
    flac = write_flac(tmp_path / "vi-1_1.flac", sine_raw(W)[:, ::-1].copy())  # canaux inversés
    rep = {}
    compute_features(flac, CAL, report=rep)
    assert rep["mapping_plausible"] is False


def test_original_flac_is_never_modified(tmp_path):
    flac = write_flac(tmp_path / "vi-1_1.flac", sine_raw(W))
    before = hashlib.sha256(flac.read_bytes()).hexdigest()
    compute_features(flac, CAL, blocksize=3_000, report={})
    assert hashlib.sha256(flac.read_bytes()).hexdigest() == before
