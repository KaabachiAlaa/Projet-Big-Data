# Règles de qualité et de traitement (J2)

Document de référence pour la version Python. La version Spark (J3) doit appliquer
exactement les mêmes règles. Ces règles reprennent celles fixées en J1 (section 4 de la
fiche d'audit) et précisent les choix faits à l'implémentation.

## 1. Règles communes

| Règle | Application |
|---|---|
| Aucune ligne ni aucun échantillon supprimé automatiquement | Seules exceptions : une ligne dont le timestamp est illisible ne peut pas être placée dans une fenêtre ; elle est comptée dans `rows_bad_timestamp` du rapport de qualité |
| Les trous restent visibles | Une fenêtre vide n'est jamais remplie ni interpolée (traitement A : aucune ligne ; traitement B : ligne explicite `is_missing_window = True`) |
| Écart-type de population | Diviseur n (ddof = 0), dans pandas, NumPy et Spark |
| Données originales intactes | Lecture seule ; les sorties vont uniquement dans `results/` (testé par somme de contrôle) |
| Blocs | Lecture par blocs ; on ne moyenne jamais des moyennes de blocs : chaque bloc fournit (n, moyenne, M2, max) combinés par la formule de Chan, ou des sommes partielles pour UK-DALE |
| MetroPT-3 et UK-DALE | Jamais fusionnés en un jeu unique |
| Tolérance d'équivalence | `|v - r| <= 1e-9 + 1e-9 * |r|`, `r` = référence Python, fixée avant toute comparaison |
| Comparaison | Après tri par clé ; mêmes clés, même nombre de lignes, valeurs proches |

## 2. MetroPT-3 : traitement A

- Fenêtres fixes de 5 minutes alignées sur l'horloge : `window_start = floor(timestamp, 5 min)` ; borne basse incluse (10:05:00 appartient à la fenêtre 10:05).
- Par fenêtre : `n` (nombre de lignes), puis pour chaque variable `_mean`, `_max`, `_std_pop`.
- Variables par défaut : les 7 analogiques (TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current). Les 8 signaux d'état 0/1 sont optionnels (`--all-variables`).
- Timestamps lus mois/jour/année si le format est `M/D/YYYY` (cohérent avec le début au 2020-02-01 de J1) ; le format retenu est consigné dans le rapport.
- Valeurs non numériques ou vides : traitées comme manquantes pour cette variable uniquement ; la ligne reste comptée dans `n` ; le nombre est consigné (`invalid_or_missing`).
- Pressions légèrement négatives (TP2 min -0,032 bar) et températures extrêmes : conservées.
- Doublons de timestamps : conservés (J1 en a trouvé 0), comptés dans `adjacent_duplicate_timestamps`.
- Lignes dans le désordre : résultat identique (les fenêtres ne dépendent pas de l'ordre), nombre consigné dans `non_monotonic_steps`.
- Une fenêtre sans observation n'apparaît pas dans `metropt3_features.csv`.
- Unités (J1, section 2.5) : TP2, TP3, H1, DV_pressure, Reservoirs en bar ; Oil_temperature en °C ; Motor_current en A ; 8 signaux d'état 0/1. Les moyennes, maxima et écarts-types gardent l'unité de la variable.
- Fuseau horaire : les horodatages du CSV n'ont pas de fuseau ; ils sont utilisés tels quels, sans conversion. Les fenêtres sont donc alignées sur l'heure écrite dans le fichier.
- Moyenne = moyenne des observations, non pondérée par le temps (le guide, section 8, demande de distinguer les deux). Le pas est irrégulier (J1 : 8 à 13 s en général, trous jusqu'à environ 48 h) ; la variante pondérée par le temps n'est pas calculée.
- `n` = nombre de lignes de la fenêtre (tous signaux confondus). L'effectif utile d'une variable vaut `n` moins ses valeurs invalides : les deux coïncident ici puisque J1 ne trouve aucune valeur manquante. À préciser si une version du jeu en contient.
- Distinction posée en J1 (section 4.1) : **erreur de mesure** (valeur non numérique, vide ou hors domaine {0, 1} pour un signal d'état) → exclue pour cette seule variable et consignée ; **absence de donnée** (trou dans le temps) → laissée visible ; **observation inhabituelle** (température extrême, petite pression négative) → conservée.
- Les valeurs exclues sont consignées avec leur motif dans `excluded_values` du rapport de qualité : numéro de ligne de données, colonne, valeur brute, motif. Le journal détaillé est limité aux 1 000 premières ; `excluded_total` donne le total exact.

## 3. UK-DALE : traitement A

- Fenêtres fixes de 1 s = 16 000 échantillons, numérotées depuis le début du fichier : fenêtre k = échantillons [16000·k, 16000·(k+1)).
- Lecture du FLAC en `int32` (échelle de libsndfile pour du PCM 24 bits : valeur × 256). Les coefficients de calibration de la maison 1 sont cohérents avec cette échelle.
- Calibration appliquée avant tout calcul : tension (V) = canal 0 × `volts_per_adc_step` ; courant (A) = canal 1 × `amps_per_adc_step`.
- Par fenêtre : `i_rms = sqrt(Σ I² / n)`, `u_rms = sqrt(Σ U² / n)`, `p_mean = Σ U·I / n`.
- Dernière fenêtre incomplète conservée avec son effectif réel (`n_samples`) ; `is_complete` la signale.
- Association des canaux : canal 0 = tension, canal 1 = courant, contrôle de plausibilité (U_rms global entre 150 et 300 V) consigné dans `mapping_plausible`.
- Le fichier doit être à 16 000 Hz et à 2 canaux, sinon le traitement s'arrête.
- `phase_difference` : consignée dans le rapport, **non appliquée** (le guide ne prévoit que la mise à l'échelle).
- Début du fichier : convention J1, secondes Unix du nom `vi-<unix>_<suffixe>.flac`, suffixe ignoré.
- Unités après calibration : tension en V, courant en A, puissance en W. Les horodatages calculés sont en UTC (temps Unix).
- Les échantillons sont réguliers (pas de 1/16 000 s) : moyenne des observations et moyenne pondérée par le temps coïncident.
- Portée : un seul fichier FLAC à la fois (extrait d'une heure, comme en J1). Les fenêtres à cheval sur deux fichiers successifs ne sont pas gérées ; la continuité entre fichiers reste à contrôler lors de la constitution des lots (J1, section 3.2).

## 4. Traitement B : variables temporelles

- Entrée : les résultats du traitement A ; la série est replacée sur la grille régulière complète des fenêtres.
- Fenêtre absente : ligne conservée, `is_missing_window = True`, `n` = 0, indicateurs vides.
- `x_variation = x_t - x_(t-1)` ; vide si la fenêtre courante ou la précédente est absente (`prev_window_missing`).
- `x_hist_mean` : moyenne sur la durée précédente (défaut 5 min), pondérée par le nombre de mesures de chaque fenêtre ; seules les fenêtres strictement antérieures sont utilisées. L'historique est défini en durée, jamais en nombre de lignes.
- `x_vs_hist = x_t - x_hist_mean`.
- `hist_n_windows` et `hist_complete` indiquent combien de fenêtres de l'historique existent réellement.
- Indicateurs : `*_mean` pour MetroPT-3 ; `i_rms`, `u_rms`, `p_mean` pour UK-DALE.

## 5. Fichiers produits

| Fichier | Contenu |
|---|---|
| `results/metropt3/metropt3_features.csv` | Traitement A MetroPT-3 |
| `results/metropt3/metropt3_quality_report.json` | Compteurs de qualité de la lecture |
| `results/ukdale/ukdale_features.csv` | Traitement A UK-DALE |
| `results/ukdale/ukdale_quality_report.json` | Contrôles du signal, calibration utilisée |
| `results/metropt3/metropt3_datamining.csv` | Traitement B MetroPT-3 (exemple) |
| `results/ukdale/ukdale_datamining.csv` | Traitement B UK-DALE |

## 6. Points ouverts à valider

1. **Historique de MetroPT-3.** Avec des fenêtres de 5 min, « les 5 minutes précédentes » se réduit à la fenêtre précédente : `_hist_mean` est alors égal à la valeur précédente. Durée à confirmer d'après le guide (option `--history-minutes`).
2. **Historique de UK-DALE.** 5 min = 300 fenêtres d'1 s, alors que le fichier dure environ une heure : les 300 premières lignes ont un historique incomplet (`hist_complete = False`).
3. **`phase_difference`.** À appliquer ou non selon le cours.
4. **Suffixe du nom FLAC** (`_236972`) : probablement des microsecondes, ignoré pour rester cohérent avec J1.
5. **Origine du FLAC** (`house_1`) : à confirmer, sinon la calibration ne s'applique pas (J1 section 5).
6. **Format réel des timestamps MetroPT-3** (avec ou sans secondes) : à relire dans le rapport de qualité après exécution.

## 7. Rapports de qualité : champs

`metropt3_quality_report.json` : `rows_read` (lignes lues), `rows_bad_timestamp` (lignes au timestamp illisible, exclues),
`non_monotonic_steps` (retours en arrière du temps), `adjacent_duplicate_timestamps` (timestamps identiques consécutifs),
`invalid_or_missing` (par variable : valeurs vides ou non numériques), `binary_out_of_domain` (par signal d'état : valeurs hors {0, 1}),
`timestamp_format`, `first_timestamp`, `last_timestamp`, `windows` (nombre de fenêtres produites),
`excluded_total`, `excluded_values` (ligne, colonne, valeur, motif), `excluded_logged_limit`.

`ukdale_quality_report.json` : `file`, `sample_rate`, `window_size`, `blocksize`, `max_seconds`, `total_samples`, `windows`,
`incomplete_windows`, `last_window_samples`, `channel_mapping`, `global_u_rms_v`, `global_i_rms_a`, `mapping_plausible`,
`calibration` (coefficients lus), `phase_difference_applied` (toujours faux), `start_unix_from_name`.

### Valeurs attendues sur les vrais fichiers (d'après J1)

| Rapport | Champ | Attendu |
|---|---|---|
| MetroPT-3 | `rows_read` | 1 516 948 |
| MetroPT-3 | `rows_bad_timestamp`, `adjacent_duplicate_timestamps`, `non_monotonic_steps` | 0 |
| MetroPT-3 | `invalid_or_missing` (toutes variables) | 0 |
| MetroPT-3 | `first_timestamp` / `last_timestamp` | 2020-02-01 00:00:00 / 2020-09-01 03:59:50 |
| UK-DALE | `total_samples` (sans `--max-seconds`) | 57 593 173 |
| UK-DALE | `windows` / `incomplete_windows` / `last_window_samples` | 3 600 / 1 / 9 173 |
| UK-DALE | `global_u_rms_v` / `global_i_rms_a` | environ 243,3 V / 1,06 A |
| UK-DALE | `mapping_plausible` | vrai |

Tout écart avec ce tableau est à expliquer avant de passer à l'étape 6.

## 8. Versions

Environnement de J1 : Windows 11, Python 3.14, pandas 2.3.3, numpy 2.3.5, soundfile 0.14.0, pyarrow 26.0.1, PySpark 4.3.0, Java Temurin 17.0.21.
Versions avec lesquelles les 71 tests ont été exécutés : Python 3.13, pandas 2.3.3 et 3.0.5, numpy 2.3.5, soundfile 0.14.0 (Python 3.14 non testé).
À compléter avec la sortie de `pip freeze` de l'environnement qui produit les résultats finaux.

## 9. Correspondance avec le guide (jalon J2 : « valeurs vérifiées et règles de qualité explicites »)

| Exigence du guide | Où | Statut |
|---|---|---|
| Traitement A MetroPT-3 : fenêtres de 5 min, effectif, moyenne, maximum, écart-type population (§8) | `metropt3_reference.py` | fait |
| Traitement A UK-DALE : fenêtres d'1 s, courant efficace, puissance active moyenne (§8) ; la tension efficace est ajoutée | `ukdale_reference.py` | fait |
| Même durée de fenêtre et même écart-type (diviseur n) dans toutes les méthodes (§8) | règles 1 à 3 | à reprendre tel quel en J3 |
| Traitement B : variation et moyenne historique, fenêtre manquante identifiable (§8) | `datamining_features.py` | fait |
| Calcul par blocs, fusion numériquement stable de la variance, cas nul ou à une observation (§9) | `common.py` | fait, testé |
| Tests : borne exacte, dernière fenêtre incomplète, partitions inégales, doublons, valeurs invalides, intervalle vide, mêmes clés et valeurs proches (§9) | `tests/` | fait |
| Test : fenêtre coupée entre deux **fichiers** (§9) | non couvert pour UK-DALE (un seul fichier) | à faire pour les lots multi-fichiers |
| Tolérance figée avant comparaison, règle explicite pour les valeurs manquantes (§9) | règle 1 ; NaN = NaN | fait |
| Règles de qualité : unités, fuseau, fenêtres, valeurs exclues et leur motif (§7, étape 2) | sections 1 à 3 et 7 | fait |
| Exemples numériques du guide (§8 : √10 et 660 W ; §9 : moyenne 64) | `tests/` | fait |
| Valeurs vérifiées sur les **vrais** fichiers | tableau de la section 7 | **à faire** : lancer les scripts et comparer |
| Versions effectivement validées figées (§6, §11) | `requirements.txt` | versions figées ; `pip freeze` final à ajouter |
