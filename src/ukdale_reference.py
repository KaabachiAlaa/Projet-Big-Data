"""Référence Python UK-DALE : Traitement A (fenêtres fixes d'1 seconde).

Pour chaque fenêtre de 16 000 échantillons : courant efficace (A), tension
efficace (V) et puissance active moyenne (W).

    I_rms = sqrt(sum(I_i^2) / n)    U_rms = sqrt(sum(U_i^2) / n)    P_moy = sum(U_i * I_i) / n

Règles (audit J1) :
- calibration appliquée AVANT tout calcul (tension en V, courant en A) ;
- canal 0 = tension, canal 1 = courant (plausibilité vérifiée et consignée) ;
- lecture par blocs, sans charger le fichier entier ;
- fenêtres alignées sur le début du fichier : fenêtre k = échantillons
  [k*16000, (k+1)*16000) ; une fenêtre coupée entre deux blocs est reconstituée
  en additionnant des sommes partielles ;
- la dernière fenêtre incomplète est conservée avec son effectif réel (n_samples).

Lecture FLAC : dtype="int32". Pour du PCM 24 bits, libsndfile renvoie alors les
échantillons décalés de 8 bits (valeur 24 bits x 256). Les coefficients du
fichier de calibration (volts_per_adc_step ~ 1,9e-7 : pleine échelle ~ 408 V
crête) sont cohérents avec cette échelle, d'où la lecture en int32 brut.

phase_difference (calibration) n'est PAS appliquée : le guide ne prévoit que la
mise à l'échelle en volts et en ampères ; le décalage de phase est consigné.

Usage :
    python -m src.ukdale_reference --input data/ukdale/vi-1363392000_236972.flac \\
        --calibration data/ukdale/calibration_house_1.cfg
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .common import (
    RESULTS_DIR,
    UKDALE_SAMPLE_RATE,
    UKDALE_WINDOW_SIZE,
    ensure_dirs,
)

VOLTAGE_CHANNEL = 0
CURRENT_CHANNEL = 1
# Tension secteur plausible (UK : ~230 V) : sert au contrôle d'association des canaux.
PLAUSIBLE_U_RMS = (150.0, 300.0)

REQUIRED_CAL = ("volts_per_adc_step", "amps_per_adc_step")


def read_calibration(path: str | Path) -> dict:
    """Lit calibration_house_X.cfg (lignes `clé = valeur`) sans supposer de section."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cal = {
        k: float(v)
        for k, v in re.findall(r"^\s*(\w+)\s*[=:]\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*$", text, re.M)
    }
    missing = [k for k in REQUIRED_CAL if k not in cal]
    if missing:
        raise ValueError(f"Coefficients de calibration manquants : {missing}")
    return cal


def start_unix_from_name(path: str | Path) -> int | None:
    """Premier nombre du nom `vi-<unix>_<suffixe>.flac` (convention de l'audit J1 :
    secondes Unix, suffixe ignoré)."""
    m = re.match(r"vi-(\d+)_", Path(path).name)
    return int(m.group(1)) if m else None


def iter_flac_blocks(path: str | Path, blocksize: int, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Blocs int32 de forme (n, 2) ; vérifie fréquence et nombre de canaux."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        if f.samplerate != UKDALE_SAMPLE_RATE:
            raise ValueError(f"Fréquence {f.samplerate} Hz au lieu de {UKDALE_SAMPLE_RATE} Hz")
        if f.channels != 2:
            raise ValueError(f"{f.channels} canal(aux) au lieu de 2")
        remaining = max_frames
        while True:
            n = blocksize if remaining is None else min(blocksize, remaining)
            if n <= 0:
                break
            block = f.read(n, dtype="int32", always_2d=True)
            if len(block) == 0:
                break
            if remaining is not None:
                remaining -= len(block)
            yield block


def accumulate_blocks(
    blocks: Iterable[np.ndarray],
    cal: dict,
    window_size: int = UKDALE_WINDOW_SIZE,
    voltage_channel: int = VOLTAGE_CHANNEL,
    current_channel: int = CURRENT_CHANNEL,
) -> pd.DataFrame:
    """Sommes partielles par fenêtre (n, sum_i2, sum_u2, sum_ui) puis agrégées.
    Les blocs peuvent avoir n'importe quelle taille et couper une fenêtre."""
    start = 0
    parts = []
    for blk in blocks:
        m = len(blk)
        if m == 0:
            continue
        u = blk[:, voltage_channel].astype(np.float64) * cal["volts_per_adc_step"]
        i = blk[:, current_channel].astype(np.float64) * cal["amps_per_adc_step"]
        w = (start + np.arange(m)) // window_size
        k = w - w[0]
        nb = int(k[-1]) + 1
        parts.append(
            pd.DataFrame(
                {
                    "window_index": w[0] + np.arange(nb),
                    "n_samples": np.bincount(k, minlength=nb),
                    "sum_i2": np.bincount(k, weights=i * i, minlength=nb),
                    "sum_u2": np.bincount(k, weights=u * u, minlength=nb),
                    "sum_ui": np.bincount(k, weights=u * i, minlength=nb),
                }
            )
        )
        start += m
    if not parts:
        raise ValueError("Aucun échantillon lu.")
    return pd.concat(parts, ignore_index=True).groupby("window_index", as_index=False).sum()


def finalize(sums: pd.DataFrame, window_size: int = UKDALE_WINDOW_SIZE,
             start_unix: int | None = None) -> pd.DataFrame:
    """Indicateurs par fenêtre à partir des sommes partielles agrégées."""
    n = sums["n_samples"].astype("float64")
    out = pd.DataFrame(
        {
            "window_index": sums["window_index"].astype("int64"),
            "n_samples": sums["n_samples"].astype("int64"),
            "is_complete": sums["n_samples"] == window_size,
            "i_rms": np.sqrt(sums["sum_i2"] / n),
            "u_rms": np.sqrt(sums["sum_u2"] / n),
            "p_mean": sums["sum_ui"] / n,
        }
    )
    out.insert(1, "window_start_s", out["window_index"] * (window_size / UKDALE_SAMPLE_RATE))
    if start_unix is not None:
        out.insert(
            2,
            "window_start_utc",
            pd.to_datetime(start_unix + out["window_start_s"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"),
        )
    return out.sort_values("window_index").reset_index(drop=True)


def compute_features(
    flac_path: str | Path,
    cal: dict,
    blocksize: int = 10 * UKDALE_SAMPLE_RATE,
    max_seconds: float | None = None,
    window_size: int = UKDALE_WINDOW_SIZE,
    report: dict | None = None,
) -> pd.DataFrame:
    """Traitement A complet sur un FLAC : lecture par blocs, calibration, indicateurs par fenêtre.
    Si `report` est fourni (dict), il est rempli avec les contrôles de qualité."""
    max_frames = None if max_seconds is None else int(max_seconds * UKDALE_SAMPLE_RATE)
    sums = accumulate_blocks(iter_flac_blocks(flac_path, blocksize, max_frames), cal, window_size)
    feats = finalize(sums, window_size, start_unix_from_name(flac_path))

    if report is not None:
        total = int(sums["n_samples"].sum())
        u_glob = float(np.sqrt(sums["sum_u2"].sum() / total))
        i_glob = float(np.sqrt(sums["sum_i2"].sum() / total))
        report.update(
            file=Path(flac_path).name,
            sample_rate=UKDALE_SAMPLE_RATE,
            window_size=window_size,
            blocksize=blocksize,
            max_seconds=max_seconds,
            total_samples=total,
            windows=len(feats),
            incomplete_windows=int((~feats["is_complete"]).sum()),
            last_window_samples=int(feats["n_samples"].iloc[-1]),
            channel_mapping={"voltage": VOLTAGE_CHANNEL, "current": CURRENT_CHANNEL},
            global_u_rms_v=u_glob,
            global_i_rms_a=i_glob,
            mapping_plausible=bool(PLAUSIBLE_U_RMS[0] <= u_glob <= PLAUSIBLE_U_RMS[1]),
            calibration=cal,
            phase_difference_applied=False,
            start_unix_from_name=start_unix_from_name(flac_path),
        )
    return feats


def compute_features_numpy(raw: np.ndarray, cal: dict, window_size: int = UKDALE_WINDOW_SIZE) -> pd.DataFrame:
    """Calcul direct en mémoire, fenêtre par fenêtre avec np.mean (autre ordre de
    sommation que la version par blocs) : contrôle croisé pour les tests.
    `raw` : tableau (N, 2) d'entiers bruts (canal 0 tension, canal 1 courant)."""
    rows = []
    for k, s in enumerate(range(0, len(raw), window_size)):
        seg = raw[s : s + window_size]
        u = seg[:, VOLTAGE_CHANNEL].astype(np.float64) * cal["volts_per_adc_step"]
        i = seg[:, CURRENT_CHANNEL].astype(np.float64) * cal["amps_per_adc_step"]
        rows.append(
            {
                "window_index": k,
                "n_samples": len(seg),
                "i_rms": float(np.sqrt(np.mean(i**2))),
                "u_rms": float(np.sqrt(np.mean(u**2))),
                "p_mean": float(np.mean(u * i)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Point d'entrée en ligne de commande (voir --help)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="fichier FLAC UK-DALE (lu en lecture seule)")
    ap.add_argument("--calibration", required=True, help="calibration_house_X.cfg")
    ap.add_argument("--output", default=str(RESULTS_DIR / "ukdale" / "ukdale_features.csv"))
    ap.add_argument("--blocksize", type=int, default=10 * UKDALE_SAMPLE_RATE, help="échantillons par bloc")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="ne traiter que les N premières secondes (extrait du signal)")
    args = ap.parse_args()

    ensure_dirs()
    cal = read_calibration(args.calibration)
    report: dict = {}
    feats = compute_features(args.input, cal, args.blocksize, args.max_seconds, report=report)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    (out.parent / "ukdale_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(feats)} fenêtres -> {out}")
    if not report["mapping_plausible"]:
        print(f"ATTENTION : U_rms global = {report['global_u_rms_v']:.1f} V, "
              "association des canaux à vérifier (--calibration / maison d'origine).")


if __name__ == "__main__":
    main()
