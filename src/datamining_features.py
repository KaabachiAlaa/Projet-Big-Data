"""Traitement B : variables temporelles pour le datamining.

À partir des résultats du traitement A (fenêtres), on calcule pour chaque
indicateur x (ex. TP2_mean, i_rms, p_mean) :

- x_variation  = x_t - x_(t-1)   (fenêtre immédiatement précédente) ;
- x_hist_mean  = moyenne de x sur les fenêtres de la durée précédente
                 (par défaut 5 minutes), pondérée par le nombre de mesures de
                 chaque fenêtre (pas une moyenne de moyennes non pondérée) ;
- x_vs_hist    = x_t - x_hist_mean (comparaison de la valeur à son historique).

Règles (audit J1 et guide) :
- la série est replacée sur la grille régulière complète des fenêtres : une
  fenêtre absente reste une ligne identifiable (is_missing_window = True, n = 0,
  indicateurs vides) ; elle n'est ni supprimée ni remplie ni interpolée ;
- l'historique se définit en DURÉE (nombre de fenêtres de la grille), jamais en
  nombre de lignes : les trous ne décalent donc pas l'historique ;
- variation vide si la fenêtre courante ou la précédente est absente ;
- hist_mean n'utilise que des fenêtres strictement antérieures (aucune fuite du
  futur) ; hist_n_windows et hist_complete indiquent la qualité de l'historique.

Usage :
    python -m src.datamining_features --dataset ukdale
    python -m src.datamining_features --dataset metropt3 [--history-minutes 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .common import METROPT3_WINDOW, RESULTS_DIR, UKDALE_SAMPLE_RATE, UKDALE_WINDOW_SIZE


def trailing_sum(a: np.ndarray, h: int) -> np.ndarray:
    """Somme des h valeurs STRICTEMENT avant chaque position (début complété par 0)."""
    padded = np.concatenate([np.zeros(h), np.asarray(a, dtype=float)])
    return sliding_window_view(padded, h)[: len(a)].sum(axis=1)


def temporal_features(
    feats: pd.DataFrame,
    pos: pd.Series,
    indicators: list[str],
    weight_col: str,
    history_windows: int,
) -> pd.DataFrame:
    """Ajoute variation / historique / écart à l'historique sur la grille complète.

    feats : tableau du traitement A ; pos : position entière de chaque fenêtre
    sur la grille régulière (0, 1, 2, ... sans trou si aucune fenêtre manque).
    Retourne le tableau reindexé sur la grille, colonne `_pos` en première colonne.
    """
    if history_windows < 1:
        raise ValueError("history_windows doit être >= 1")
    if pos.duplicated().any():
        raise ValueError("Positions de fenêtres dupliquées : clés non uniques")

    d = feats.assign(_pos=pos.to_numpy()).sort_values("_pos")
    grid = pd.RangeIndex(int(d["_pos"].min()), int(d["_pos"].max()) + 1, name="_pos_grid")
    d = d.set_index("_pos").reindex(grid)
    d.index.name = "_pos"

    present = d[weight_col].notna()
    d["is_missing_window"] = ~present
    d[weight_col] = d[weight_col].fillna(0).astype("int64")
    d["prev_window_missing"] = ~present.shift(1, fill_value=False)
    d.loc[d.index[0], "prev_window_missing"] = True  # pas de fenêtre précédente

    for c in indicators:
        x = d[c]
        valid = x.notna()
        w = d[weight_col].where(valid, 0).to_numpy(dtype=float)
        num = trailing_sum(np.where(valid, x, 0.0) * w, history_windows)
        den = trailing_sum(w, history_windows)
        hist = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=d.index)
        d[f"{c}_variation"] = x - x.shift(1)          # vide si t ou t-1 est absente
        d[f"{c}_hist_mean"] = hist
        d[f"{c}_vs_hist"] = x - hist

    d["hist_n_windows"] = trailing_sum(present.to_numpy(dtype=float), history_windows).astype("int64")
    d["hist_complete"] = d["hist_n_windows"] == history_windows
    return d.reset_index()


def _history_windows(history_minutes: float, window_seconds: float) -> int:
    h = history_minutes * 60 / window_seconds
    if h < 1 or abs(h - round(h)) > 1e-9:
        raise ValueError(
            f"Historique de {history_minutes} min incompatible avec des fenêtres de {window_seconds:g} s"
        )
    return int(round(h))


def metropt3_datamining(feats: pd.DataFrame, history_minutes: float = 5.0,
                        variables: list[str] | None = None) -> pd.DataFrame:
    """Traitement B sur les fenêtres MetroPT-3 (colonnes `window_start`, `n`, `<variable>_mean`)."""
    feats = feats.copy()
    feats["window_start"] = pd.to_datetime(feats["window_start"])
    step = pd.Timedelta(METROPT3_WINDOW)
    origin = feats["window_start"].min()
    ratio = (feats["window_start"] - origin) / step
    if not np.allclose(ratio, np.round(ratio)):
        raise ValueError("Fenêtres non alignées sur la grille de 5 minutes")
    pos = pd.Series(np.round(ratio).astype("int64"), index=feats.index)

    cols = [f"{v}_mean" for v in variables] if variables else [c for c in feats.columns if c.endswith("_mean")]
    h = _history_windows(history_minutes, step.total_seconds())
    out = temporal_features(feats, pos, cols, "n", h)
    out["window_start"] = origin + out["_pos"] * step
    first = ["window_start", "is_missing_window", "n"]
    return out.drop(columns="_pos")[first + [c for c in out.columns if c not in first + ["_pos"]]]


def ukdale_datamining(feats: pd.DataFrame, history_minutes: float = 5.0) -> pd.DataFrame:
    """Traitement B sur les fenêtres UK-DALE (colonnes `window_index`, `n_samples`, `i_rms`, `u_rms`, `p_mean`)."""
    window_seconds = UKDALE_WINDOW_SIZE / UKDALE_SAMPLE_RATE
    h = _history_windows(history_minutes, window_seconds)
    out = temporal_features(feats, feats["window_index"], ["i_rms", "u_rms", "p_mean"], "n_samples", h)
    out["window_index"] = out["_pos"].astype("int64")
    out["window_start_s"] = out["window_index"] * window_seconds
    if "window_start_utc" in feats.columns:
        first = feats.sort_values("window_index").iloc[0]
        t0 = pd.Timestamp(first["window_start_utc"]) - pd.Timedelta(seconds=first["window_start_s"])
        out["window_start_utc"] = (t0 + pd.to_timedelta(out["window_start_s"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["is_complete"] = out["is_complete"].eq(True)  # fenêtres manquantes : False
    front = [c for c in ["window_index", "window_start_s", "window_start_utc",
                         "is_missing_window", "n_samples", "is_complete"] if c in out.columns]
    return out.drop(columns="_pos")[front + [c for c in out.columns if c not in front + ["_pos"]]]


def main() -> None:
    """Point d'entrée en ligne de commande (voir --help)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["metropt3", "ukdale"])
    ap.add_argument("--input", help="CSV du traitement A (défaut : results/<dataset>/<dataset>_features.csv)")
    ap.add_argument("--output", help="CSV de sortie (défaut : results/<dataset>/<dataset>_datamining.csv)")
    ap.add_argument("--history-minutes", type=float, default=5.0, help="durée de l'historique (défaut : 5)")
    ap.add_argument("--variables", help="MetroPT-3 : variables séparées par des virgules (défaut : toutes)")
    args = ap.parse_args()

    ds = args.dataset
    src = Path(args.input) if args.input else RESULTS_DIR / ds / f"{ds}_features.csv"
    dst = Path(args.output) if args.output else RESULTS_DIR / ds / f"{ds}_datamining.csv"
    feats = pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if ds == "metropt3":
        variables = args.variables.split(",") if args.variables else None
        out = metropt3_datamining(feats, args.history_minutes, variables)
        out.to_csv(dst, index=False, date_format="%Y-%m-%d %H:%M:%S")
    else:
        out = ukdale_datamining(feats, args.history_minutes)
        out.to_csv(dst, index=False)
    print(f"{len(out)} fenêtres ({int(out['is_missing_window'].sum())} manquantes) -> {dst}")


if __name__ == "__main__":
    main()
