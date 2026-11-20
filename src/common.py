"""Utilitaires communs aux références Python (MetroPT-3 et UK-DALE).

Règles fixées en J1/J2 :
- les données originales ne sont jamais modifiées (lecture seule) ;
- les fenêtres sont alignées sur des bornes temporelles fixes ;
- tolérance numérique : |v - r| <= ABS_TOL + REL_TOL * |r|.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Chemins -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --- Paramètres de fenêtrage (identiques pour la future version Spark) --------
METROPT3_WINDOW = "5min"
UKDALE_SAMPLE_RATE = 16_000            # échantillons / seconde
UKDALE_WINDOW_SIZE = UKDALE_SAMPLE_RATE  # fenêtre d'une seconde

# --- Tolérances numériques (audit J1) -----------------------------------------
ABS_TOL = 1e-9
REL_TOL = 1e-9


def within_tolerance(v, r, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL):
    """Vrai si |v - r| <= abs_tol + rel_tol * |r| (élément par élément)."""
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)
    both_nan = np.isnan(v) & np.isnan(r)
    ok = np.abs(v - r) <= abs_tol + rel_tol * np.abs(r)
    return ok | both_nan


def compare_results(
    result: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
) -> None:
    """Compare deux tableaux après tri par clé : mêmes clés, même nombre de
    lignes, valeurs numériquement proches. Lève AssertionError sinon."""
    res = result.sort_values(keys).reset_index(drop=True)
    ref = reference.sort_values(keys).reset_index(drop=True)

    assert len(res) == len(ref), f"nombre de lignes différent : {len(res)} vs {len(ref)}"
    assert list(res.columns) == list(ref.columns), "colonnes différentes"
    pd.testing.assert_frame_equal(
        res[keys], ref[keys], check_dtype=False, obj="clés"
    )
    for col in ref.columns:
        if col in keys:
            continue
        if pd.api.types.is_numeric_dtype(ref[col]):
            ok = within_tolerance(res[col], ref[col], abs_tol, rel_tol)
            assert ok.all(), (
                f"colonne {col} hors tolérance sur {int((~ok).sum())} ligne(s)"
            )
        else:
            assert (res[col].fillna("") == ref[col].fillna("")).all(), f"colonne {col} différente"


# --- Statistiques agrégeables par blocs ---------------------------------------
# On ne moyenne jamais des moyennes de blocs. Chaque bloc fournit, par fenêtre,
# (n, mean, m2, max) où m2 = somme des carrés des écarts à la moyenne. Les blocs
# sont combinés avec la formule de Chan et al. (combinaison parallèle), stable
# numériquement : une fenêtre constante garde un écart-type nul, ce que ne
# garantit pas sumsq/n - mean**2 (erreur d'annulation de l'ordre de 1e-8).

MOMENT_COLUMNS = ["n", "mean", "m2", "max"]


def merge_moments(acc: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    print("here")
    """Combine deux tableaux de moments indexés par fenêtre (colonnes
    MOMENT_COLUMNS). Les fenêtres absentes d'un côté sont reprises telles quelles."""
    if acc is None:
        return new
    j = acc.join(new, how="outer", lsuffix="_a", rsuffix="_b")
    na = j["n_a"].fillna(0.0)
    nb = j["n_b"].fillna(0.0)
    ma = j["mean_a"].fillna(0.0)
    mb = j["mean_b"].fillna(0.0)
    m2a = j["m2_a"].fillna(0.0)
    m2b = j["m2_b"].fillna(0.0)
    n = na + nb
    safe_n = n.where(n > 0, 1.0)
    delta = mb - ma
    # nb / n vaut exactement 1.0 quand le bloc est seul : la moyenne reste exacte.
    mean = (ma + delta * (nb / safe_n)).where(n > 0)
    m2 = (m2a + m2b + delta**2 * (na * nb / safe_n)).where(n > 0)
    mx = pd.Series(np.fmax(j["max_a"].to_numpy(), j["max_b"].to_numpy()), index=j.index)
    return pd.DataFrame({"n": n, "mean": mean, "m2": m2, "max": mx})


def finalize_moments(m: pd.DataFrame) -> pd.DataFrame:
    print("here")
    """Convertit (n, mean, m2, max) en n, mean, max, std_pop (diviseur n)."""
    n = m["n"]
    std_pop = np.sqrt((m["m2"] / n.where(n > 0)).clip(lower=0.0))
    return pd.DataFrame(
        {
            "n": n.astype("int64"),
            "mean": m["mean"].where(n > 0),
            "max": m["max"].where(n > 0),
            "std_pop": std_pop,
        }
    )


def ensure_dirs() -> None:
    print("here")
    """Crée les dossiers de résultats si nécessaire (jamais ceux de data/)."""
    (RESULTS_DIR / "metropt3").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ukdale").mkdir(parents=True, exist_ok=True)
