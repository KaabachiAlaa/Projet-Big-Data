# BigData_Project : référence Python (J2)

Référence Python pour les traitements A et B sur **MetroPT-3** (fenêtres de 5 min)
et **UK-DALE** (fenêtres d'1 s = 16 000 échantillons). Elle sert de vérité de
comparaison pour la version Spark (J3) : mêmes fenêtres, mêmes clés, mêmes tolérances.

## Structure

```
BigData_Project/
├── README.md
├── audit_rules.md              règles de qualité et de traitement
├── pytest.ini
├── requirements.txt
├── data/                       données originales (lecture seule, jamais modifiées)
│   ├── metropt3/               CSV MetroPT-3
│   └── ukdale/                 FLAC + calibration_house_1.cfg
├── src/
│   ├── __init__.py             fichier vide
│   ├── common.py               tolérance, comparaison par clé, fusion stable des blocs
│   ├── metropt3_reference.py   traitement A MetroPT-3
│   ├── ukdale_reference.py     traitement A UK-DALE
│   └── datamining_features.py  traitement B (variation, moyenne historique)
├── notebooks/                  explication et restitution (appellent src/, ne le remplacent pas)
│   ├── 01_metropt3.ipynb
│   ├── 02_ukdale.ipynb
│   └── 03_verification.ipynb
├── tests/
│   ├── __init__.py             fichier vide
│   ├── test_common.py
│   ├── test_metropt3.py
│   ├── test_ukdale.py
│   └── test_datamining.py
└── results/                    CSV et rapports produits (metropt3/, ukdale/)
```

## Installation

Versions figées dans `requirements.txt` (celles de J1). Tests exécutés avec Python 3.13, pandas 2.3.3 et 3.0.5, numpy 2.3.5, soundfile 0.14.0.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Règles principales

- Données originales intactes ; les résultats vont uniquement dans `results/`.
- Lecture par blocs ; statistiques combinées via (n, moyenne, M2, max) avec la formule de
  Chan (stable ; jamais une moyenne de moyennes de blocs).
- Écart-type de population (ddof = 0).
- Les fenêtres vides ne sont pas supprimées : elles restent identifiables.
- Tolérance de comparaison : |v - r| <= 1e-9 + 1e-9 |r|.

Le détail complet est dans `audit_rules.md`.

## Lancer MetroPT-3 (traitement A)

```bash
python -m src.metropt3_reference --input "data/metropt3/MetroPT3(AirCompressor).csv"
# sorties : results/metropt3/metropt3_features.csv et metropt3_quality_report.json
# options : --all-variables (inclure aussi les 8 signaux d'état 0/1),
#           --chunksize N (lignes par bloc, défaut 200000), --output chemin.csv
```

## Lancer UK-DALE (traitement A)

```bash
python -m src.ukdale_reference --input data/ukdale/vi-1363392000_236972.flac \
    --calibration data/ukdale/calibration_house_1.cfg
# sorties : results/ukdale/ukdale_features.csv et ukdale_quality_report.json
# options : --max-seconds N (ne traiter qu'un extrait), --blocksize N (échantillons par bloc,
#           défaut 160000), --output chemin.csv
```

## Lancer le traitement B (après le traitement A)

```bash
python -m src.datamining_features --dataset ukdale
python -m src.datamining_features --dataset metropt3
# options : --history-minutes 5 (défaut), --variables TP2,Oil_temperature (MetroPT-3),
#           --input et --output pour d'autres chemins
# sorties : results/<dataset>/<dataset>_datamining.csv
```

## Notebooks

Les notebooks de `notebooks/` expliquent le travail (formules, exemples du guide, figures, rapport de qualité) en **appelant** le code de `src/` :
ils ne le remplacent pas, et les scripts et les tests restent la référence.

```bash
jupyter lab notebooks/        # ou : python -m jupyter lab
```

- `01_metropt3.ipynb` : traitements A et B sur MetroPT-3.
- `02_ukdale.ipynb` : calibration, traitements A et B sur UK-DALE.
- `03_verification.ipynb` : tolérance, comparaison par clé, exécution des 71 tests, versions.

Si les vrais fichiers (`data/`, `results/`) sont absents, les notebooks tournent sur des **données synthétiques**, signalées dans chaque figure.
Après avoir lancé les scripts sur les vrais fichiers, relancer les notebooks (Kernel > Restart & Run All) pour obtenir les vrais résultats.

## Tests de vérification (71 tests)

Ils utilisent des données synthétiques : aucun fichier réel n'est nécessaire.
À lancer depuis la racine du projet :

```bash
python -m pytest                                  # tous les tests
python -m pytest tests/test_metropt3.py           # un seul fichier
python -m pytest -k "boundary or incomplete"      # par mot-clé
```

Chaque résultat est comparé à un calcul indépendant (pandas ou NumPy), après tri par
clé : mêmes clés, même nombre de lignes, valeurs proches selon la tolérance J1.

### `tests/test_common.py` (10 tests) : outils communs

- Règle de tolérance J1 (`1e-9 + 1e-9·|r|`, écart mesuré par rapport à la référence).
- Deux NaN sont égaux ; NaN contre un nombre ne l'est pas.
- `compare_results` trie par clé avant de comparer.
- `compare_results` détecte un nombre de lignes, des clés ou des valeurs différents.
- La fusion par blocs donne le même résultat que le calcul global, pour toute coupure.
- La fusion conserve les fenêtres présentes d'un seul côté.
- Une fenêtre constante coupée en deux blocs garde un écart-type nul.
- L'écart-type est celui de la population (exemple pédagogique du plan de travail : 2,61).
- Exemple du guide (§9) : partitions [60, 62] et [64, 66, 68] → moyenne 64, et non 63,5.
- Fenêtre à une seule observation : écart-type nul, moyenne et maximum exacts.

### `tests/test_metropt3.py` (23 tests) : traitement A MetroPT-3

- Exemple pédagogique du plan de travail (absent du guide) : n = 5, moyenne 68, maximum 72, écart-type 2,61.
- Fenêtre à une seule observation : écart-type nul.
- Valeurs exclues consignées dans le rapport avec ligne, colonne, valeur et motif (guide §7).
- Fenêtre qui commence exactement sur une borne (10:05:00 ouvre la fenêtre suivante).
- Blocs de tailles différentes : 1, 2, 7, 17, 64, 1000 et 10⁶ lignes (7 cas).
- Fenêtre coupée entre deux blocs.
- Dernière fenêtre incomplète conservée avec son effectif réel.
- Périodes sans observations : aucune fenêtre inventée ni supprimée.
- Doublons : conservés et comptés, pas supprimés.
- Valeurs invalides (texte, vide) : ignorées pour cette variable seulement.
- Timestamps illisibles : lignes exclues et consignées dans le rapport.
- Lignes dans le désordre : même résultat, désordre signalé.
- Fenêtre constante : écart-type exactement nul.
- Stabilité numérique : valeurs proches de 10⁶ avec une variation de 10⁻³.
- Variable entièrement invalide dans une fenêtre : résultat vide, pas zéro.
- Formats de date ISO et mois/jour/année : résultats identiques.
- Le fichier CSV d'origine n'est jamais modifié (somme de contrôle).

### `tests/test_ukdale.py` (23 tests) : traitement A UK-DALE

- Fenêtre qui se termine exactement sur une borne : pas de fenêtre fantôme.
- Fenêtre coupée entre deux blocs, reconstituée par sommes partielles.
- Dernière fenêtre incomplète : 8 000 échantillons conservés, `is_complete = False`.
- Découpages aléatoires en blocs (5 cas) et blocs d'un seul échantillon.
- Exemple du guide (§8) : U = [220, 220] V, I = [2, 4] A → I_rms = √10 = 3,162 A, P = 660 W.
- Signal connu (230 V, 5 A, cos φ = 0,8) : 230 V, 5 A, 920 W.
- Calibration appliquée avant le calcul (facteurs 2 et 3 → 2, 3 et 6).
- Canaux inversés : tension implausible détectée.
- Lecture du fichier de calibration, et rejet d'un fichier incomplet.
- Lecture d'un vrai FLAC par blocs de 777, 12 345, 16 000 et 10⁶ échantillons (4 cas).
- Extrait avec `--max-seconds` : dernière fenêtre partielle conservée.
- Date de début déduite du nom du fichier.
- Fréquence ou nombre de canaux incorrects : rejet.
- Tension implausible signalée dans le rapport de qualité.
- Le fichier FLAC d'origine n'est jamais modifié (somme de contrôle).

### `tests/test_datamining.py` (15 tests) : traitement B

- Variation : exemple pédagogique du plan de travail (—, +3, +7, −5), absent du guide.
- Fenêtre manquante : ligne conservée, identifiable, non remplie.
- Variation vide si la fenêtre courante ou la précédente est absente.
- Historique défini en durée et non en nombre de lignes (cas avec trou).
- Moyenne historique pondérée par le nombre de mesures.
- Écart à l'historique et indicateurs de début de série (`hist_complete`).
- Aucune fuite du futur : modifier une fenêtre ne change pas le passé.
- L'ordre des lignes en entrée n'a pas d'effet ; l'entrée n'est pas modifiée.
- Clés de fenêtres dupliquées : rejet.
- Historique qui n'est pas un multiple de la fenêtre : rejet.
- Historique de 5 min sur MetroPT-3 = fenêtre précédente.
- Un indicateur vide n'est pas compté comme zéro.
- UK-DALE : trou dans `window_index` rendu explicite.
- UK-DALE : 5 minutes = 300 fenêtres.
- Positions cohérentes dans `temporal_features`.

## Avancement J2

- [x] Étape 1 : structure du projet, `common.py`, `requirements.txt`
- [~] Étape 2 : MetroPT-3 (traitement A) : code prêt et validé sur données synthétiques ; reste à lancer sur le vrai CSV
- [~] Étape 3 : UK-DALE (traitement A) : code prêt et validé sur FLAC synthétique ; reste à lancer sur le vrai fichier
- [~] Étape 4 : traitement B : code prêt et validé sur données synthétiques ; reste à lancer sur les vrais résultats du traitement A
- [x] Étape 5 : tests (`python -m pytest`, 71 tests)
- [ ] Étape 6 : comparaison et validation
- [~] Étape 7 : documentation : `audit_rules.md` écrit ; reste le `pip freeze` final et les résultats réels
