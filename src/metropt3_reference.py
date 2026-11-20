"""Référence Python MetroPT-3 : Traitement A (fenêtres fixes de 5 minutes).

Règles (audit J1) :
- fenêtres de 5 min alignées sur l'horloge (plancher du timestamp) ;
- effectif, moyenne, maximum, écart-type de population (diviseur n) ;
- aucune ligne supprimée automatiquement : seules les lignes dont le timestamp
  est illisible ne peuvent pas être fenêtrées ; elles sont comptées et consignées ;
- les périodes sans mesure ne produisent aucune ligne ni interpolation ici
  (elles sont rendues explicites dans le traitement B) ;
- lecture par blocs ; les blocs sont combinés avec des moments (n, mean, M2, max).

Usage :
    python -m src.metropt3_reference --input "data/metropt3/MetroPT3(AirCompressor).csv"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import (
    METROPT3_WINDOW,
    RESULTS_DIR,
    ensure_dirs,
    finalize_moments,
    merge_moments,
)

TS_COL = "timestamp"
ANALOG_VARS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
]
BINARY_VARS = [
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
    "Caudal_impulses",
]
# Formats essayés dans l'ordre. L'audit J1 situe le début le 2020-02-01 : un
# format "2/1/2020" se lit donc mois/jour/année.
TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]


def detect_ts_format(sample: pd.Series) -> str | None:
    """Premier format qui lit tous les timestamps de l'échantillon, sinon None."""
    sample = sample.dropna().astype(str)
    for fmt in TS_FORMATS:
        if pd.to_datetime(sample, format=fmt, errors="coerce").notna().all():
            return fmt
    return None


def parse_timestamps(series: pd.Series, fmt: str | None) -> pd.Series:
    """Convertit en datetime avec le format détecté ; valeur illisible -> NaT (jamais d'exception)."""
    if fmt is None:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    return pd.to_datetime(series.astype(str), format=fmt, errors="coerce")


def chunk_moments(df: pd.DataFrame, variables: list[str], window: str):
    """Moments par fenêtre pour un bloc : (effectif de lignes, {variable: moments})."""
    key = df[TS_COL].dt.floor(window)
    rows = key.value_counts().sort_index().astype("float64")
    out = {}
    for v in variables:
        x = df[v]
        g = x.groupby(key)
        mean = g.mean()
        dev2 = (x - x.groupby(key).transform("mean")) ** 2
        out[v] = pd.DataFrame(
            {
                "n": g.count().astype("float64"),
                "mean": mean,
                "m2": dev2.groupby(key).sum(),
                "max": g.max(),
            }
        )
    return rows, out


def assemble(rows: pd.Series, moments: dict[str, pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    """Tableau large : window_start, n (lignes), puis <var>_mean/_max/_std_pop."""
    res = pd.DataFrame({"n": rows.astype("int64")})
    for v in variables:
        f = finalize_moments(moments[v]).reindex(res.index)
        res[f"{v}_mean"] = f["mean"]
        res[f"{v}_max"] = f["max"]
        res[f"{v}_std_pop"] = f["std_pop"]
    res.index.name = "window_start"
    return res.sort_index().reset_index()


MAX_EXCLUDED_LOGGED = 1000  # plafond du journal détaillé ; le total exact reste dans excluded_total


def log_excluded(q: dict, original: pd.Series, mask: pd.Series, column: str) -> None:
    """Consigne les valeurs exclues et leur motif (guide §7, étape 2).

    `original` : valeurs avant conversion ; `mask` : True là où la conversion échoue.
    L'index d'un bloc lu par read_csv continue d'un bloc à l'autre : `row` est donc
    le numéro de ligne de données (0 = première ligne sous l'en-tête)."""
    n = int(mask.sum())
    if n == 0:
        return
    q["excluded_total"] += n
    room = MAX_EXCLUDED_LOGGED - len(q["excluded_values"])
    for idx in mask[mask].index[: max(room, 0)]:
        v = original.loc[idx]
        if pd.isna(v):
            reason = "valeur vide"
        else:
            reason = "horodatage illisible" if column == TS_COL else "valeur non numérique"
        q["excluded_values"].append(
            {"row": int(idx), "column": column, "value": None if pd.isna(v) else str(v), "reason": reason}
        )


def compute_features(
    csv_path: str | Path,
    variables: list[str] | None = None,
    chunksize: int = 200_000,
    window: str = METROPT3_WINDOW,
    report: dict | None = None,
) -> pd.DataFrame:
    """Traitement A par blocs sur le CSV MetroPT-3 (lecture seule)."""
    variables = variables or ANALOG_VARS
    usecols = [TS_COL] + list(dict.fromkeys(variables + BINARY_VARS))
    q = report if report is not None else {}
    q.update(
        rows_read=0,
        rows_bad_timestamp=0,
        non_monotonic_steps=0,
        adjacent_duplicate_timestamps=0,
        invalid_or_missing={v: 0 for v in usecols[1:]},
        binary_out_of_domain={v: 0 for v in BINARY_VARS},
        timestamp_format=None,
        excluded_total=0,
        excluded_values=[],
        excluded_logged_limit=MAX_EXCLUDED_LOGGED,
    )
    fmt, fmt_known = None, False
    prev_ts, acc_rows, acc = None, None, {v: None for v in variables}
    ts_min = ts_max = None

    for raw in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, dtype={TS_COL: str}):
        if not fmt_known:
            fmt, fmt_known = detect_ts_format(raw[TS_COL].head(1000)), True
            q["timestamp_format"] = fmt or "mixed"
        ts_original = raw[TS_COL]
        raw[TS_COL] = parse_timestamps(ts_original, fmt)
        bad = raw[TS_COL].isna()
        log_excluded(q, ts_original, bad, TS_COL)
        for c in usecols[1:]:
            num = pd.to_numeric(raw[c], errors="coerce")
            log_excluded(q, raw[c], num.isna(), c)
            q["invalid_or_missing"][c] += int(num.isna().sum())
            raw[c] = num

        q["rows_read"] += len(raw)
        q["rows_bad_timestamp"] += int(bad.sum())
        for c in BINARY_VARS:
            q["binary_out_of_domain"][c] += int((raw[c].notna() & ~raw[c].isin([0, 1])).sum())

        df = raw[~bad]  # seules les lignes sans timestamp lisible sont écartées (consignées)
        if df.empty:
            continue

        ts = df[TS_COL]
        # On ajoute le dernier timestamp du bloc précédent pour contrôler la jonction.
        ts_ext = ts if prev_ts is None else pd.concat([pd.Series([prev_ts]), ts], ignore_index=True)
        steps = ts_ext.diff().iloc[1:]
        q["non_monotonic_steps"] += int((steps < pd.Timedelta(0)).sum())
        q["adjacent_duplicate_timestamps"] += int((steps == pd.Timedelta(0)).sum())
        prev_ts = ts.iloc[-1]
        ts_min = ts.min() if ts_min is None else min(ts_min, ts.min())
        ts_max = ts.max() if ts_max is None else max(ts_max, ts.max())

        rows, moments = chunk_moments(df, variables, window)
        acc_rows = rows if acc_rows is None else acc_rows.add(rows, fill_value=0.0)
        for v in variables:
            acc[v] = merge_moments(acc[v], moments[v])

    q["first_timestamp"] = None if ts_min is None else str(ts_min)
    q["last_timestamp"] = None if ts_max is None else str(ts_max)
    if acc_rows is None:
        raise ValueError("Aucune ligne exploitable dans le fichier.")
    result = assemble(acc_rows, acc, variables)
    q["windows"] = len(result)
    return result


def compute_features_pandas(df: pd.DataFrame, variables: list[str] | None = None,
                            window: str = METROPT3_WINDOW) -> pd.DataFrame:
    """Calcul direct en mémoire (une seule passe pandas, ddof=0), indépendant du
    découpage par blocs. Sert de contrôle croisé dans les tests."""
    variables = variables or ANALOG_VARS
    d = df.copy()
    d["window_start"] = d[TS_COL].dt.floor(window)
    g = d.groupby("window_start")
    out = pd.DataFrame({"n": g.size().astype("int64")})
    for v in variables:
        out[f"{v}_mean"] = g[v].mean()
        out[f"{v}_max"] = g[v].max()
        out[f"{v}_std_pop"] = g[v].std(ddof=0)
    return out.reset_index()


def main() -> None:
    """Point d'entrée en ligne de commande (voir --help)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV MetroPT-3 (lu en lecture seule)")
    ap.add_argument("--output", default=str(RESULTS_DIR / "metropt3" / "metropt3_features.csv"))
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--all-variables", action="store_true",
                    help="inclure aussi les 8 signaux d'état 0/1 (par défaut : 7 analogiques)")
    args = ap.parse_args()

    ensure_dirs()
    variables = ANALOG_VARS + (BINARY_VARS if args.all_variables else [])
    report: dict = {}
    res = compute_features(args.input, variables, args.chunksize, report=report)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    (out.parent / "metropt3_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"{len(res)} fenêtres -> {out}")


if __name__ == "__main__":
    main()
