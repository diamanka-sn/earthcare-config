# ECA_JXBA_ACM_CLP_2B — Architecture du projet

## Structure des fichiers

```
eca_clp/
├── config.py       ← constantes, chemins HDF5, style graphique
├── io_data.py      ← téléchargement EarthCARE, lecture HDF5
├── processing.py   ← calculs (haversine, masques, grilles, temps solaire)
├── plotting.py     ← toutes les fonctions de visualisation
└── main.py         ← point d'entrée — orchestre les deux pipelines
```

## Lancement

```bash
python main.py
```

Ou dans un notebook Jupyter :
```python
%run main.py
# ou en ciblant une seule section :
from main import run_single_orbit
run_single_orbit()
```

## Modifier les paramètres

Tout passe par **`config.py`** :

| Paramètre        | Effet                                          |
|------------------|------------------------------------------------|
| `ORBIT_FRAME`    | Numéro d'orbite + frame (ex. `"09039G"`)      |
| `DATE_START/END` | Plage temporelle de recherche                  |
| `T_MIN / T_MAX`  | Fenêtre d'analyse (secondes depuis début)      |
| `LAT_REF / LON_REF` | Station de référence (actuellement Dome C)  |
| `DOWNLOAD_PERIODS` | Périodes pour le téléchargement multi-orbites|
| `CIRCLES_CONFIG` | Rayons et couleurs des cercles de distance     |

Aucun autre fichier n'a besoin d'être modifié pour changer d'orbite ou de station.

## Responsabilités par fichier

### `config.py`
- Constantes de l'expérience (orbite, dates, station, fenêtre temps)
- Dictionnaires de style (couleurs et labels des particules)
- Chemins HDF5 centralisés

### `io_data.py`
- `download_product()` / `download_multi_period()` : appels `eck.ecdownload`
- `search_product()` : wrapping `eck.search_product`
- `load_orbit()` : lecture HDF5 → dict numpy
- `load_multi_orbits()` : idem avec gestion d'erreurs
- `get_t0_utc()` : extraction de l'heure de début depuis l'en-tête

### `processing.py`
- `haversine()` / `distance_to_ref()` : calcul de distances
- `calculate_local_times()` : temps solaire local
- `mask_negative()` / `mask_particle_type()` : nettoyage des données
- `build_time_grid()` : grille T2D pour pcolormesh
- `prepare_single_orbit()` : calcule toutes les variables dérivées
- `prepare_multi_orbits()` / `build_orbit_label()` : préparation multi-orbites

### `plotting.py`
- Helpers privés (`_add_time_labels`, `_add_particle_legend`,
  `_make_polar_map`, `_add_distance_circles`)
- Fonctions publiques `plot_*` : une figure par appel, sans effet de bord
- Colormaps construites une seule fois au chargement du module

### `main.py`
- `run_single_orbit()` : télécharge → traite → affiche les 11 figures
- `run_multi_orbit()` : idem pour la synthèse multi-orbites
