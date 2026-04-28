# ECA_JXBA_ACM_CLP_2B — Architecture du projet

## Structure des fichiers

```
eca_clp/
├── config.py       ← constantes, chemins HDF5, style graphique
├── io_data.py      ← téléchargement EarthCARE, lecture HDF5 (ESA)
├── io_jaxa.py      ← lecture HDF5 (JAXA) + fusion ESA/JAXA
├── processing.py   ← calculs (haversine, masques, grilles, temps solaire)
├── gridding.py     ← accumulation sur grille lat/lon, cache NetCDF4
├── plotting.py     ← toutes les fonctions de visualisation
├── main.py         ← point d'entrée — orchestre les trois pipelines
└── data/
    └── grid_cache.nc   ← cache de la grille (généré automatiquement)
```

---

## Lancement

```bash
python main.py
```

Ou dans un notebook Jupyter :
```python
%run main.py

# Cibler une seule section :
from main import run_single_orbit, run_multi_orbit, run_grid, plot_grid_results
run_single_orbit()          # figures pour une orbite unique
run_multi_orbit()           # carte multi-orbites ESA + JAXA
grid = run_grid()           # construit/charge la grille
plot_grid_results(grid)     # cartes mean + std
```

---

## Modifier les paramètres

Tout passe par **`config.py`** — aucun autre fichier à toucher :

| Paramètre           | Effet                                           |
|---------------------|-------------------------------------------------|
| `ORBIT_FRAME`       | Numéro d'orbite + frame (ex. `"09039G"`)        |
| `DATE_START/END`    | Plage temporelle de recherche                   |
| `T_MIN / T_MAX`     | Fenêtre d'analyse (secondes depuis début)       |
| `LAT_REF / LON_REF` | Station de référence (actuellement Dome C)      |
| `DOWNLOAD_PERIODS`  | Périodes pour le téléchargement multi-orbites   |
| `CIRCLES_CONFIG`    | Rayons et couleurs des cercles de distance      |
| `GRID_PARAMS_1D`    | Paramètres scalaires accumulés dans la grille   |
| `GRID_PARAMS_2D`    | Paramètres 2D réduits verticalement puis grillés|

---

## Responsabilités par fichier

### `config.py`
- Constantes de l'expérience (orbite, dates, station, fenêtre temps)
- Dictionnaires de style (couleurs et labels des particules)
- Chemins HDF5 centralisés (`HDF5_FIELDS`, `HDF5_FIELDS_ORBIT_META`)
- Listes des paramètres à accumuler dans la grille

### `io_data.py`
- `download_product()` / `download_multi_period()` : appels `eck.ecdownload`
- `search_product()` : recherche des fichiers disponibles
- `load_orbit()` : lecture HDF5 → dict numpy
- `load_multi_orbits()` : idem avec gestion d'erreurs par fichier
- `get_t0_utc()` : extraction de l'heure UTC depuis l'en-tête HDF5

### `io_jaxa.py`
- `load_jaxa_orbits(directory)` : charge tous les `.h5` d'un dossier JAXA.
  La structure HDF5 étant identique à ESA, délègue directement à `load_orbit()`.
- `merge_orbit_sources(esa, jaxa)` : fusionne les deux listes et trie
  chronologiquement par temps absolu.

### `processing.py`
- `haversine()` / `distance_to_ref()` : distances géodésiques
- `calculate_local_times()` : temps solaire local (4 min/degré de longitude)
- `mask_negative()` / `mask_particle_type()` : nettoyage des données
- `build_time_grid()` : grille T2D pour `pcolormesh`
- `prepare_single_orbit()` : calcule toutes les variables dérivées en un appel
- `prepare_multi_orbits()` / `build_orbit_label()` : préparation multi-orbites

### `gridding.py`
Accumulation statistique sur une grille régulière `(lat × lon)`.

**Principe** : pour chaque cellule on stocke `Σx`, `Σx²` et `n` (jamais les
valeurs individuelles), ce qui permet de calculer moyenne et écart-type à la
demande sans relire les HDF5.

- `GridAccumulator(dlat=1°, dlon=10°)` : crée la grille
- `accumulate(orbit_data)` : ajoute une orbite
- `mean()` → `dict {param: ndarray (n_lat, n_lon)}`
- `std()`  → idem, écart-type avec correction de Bessel
- `count()` → nombre de mesures par cellule
- `to_dataset()` → conversion en `xr.Dataset` avec coordonnées et attributs CF
- `save(path)` : sauvegarde en **NetCDF4** (`.nc`) — inclut mean/std/count
  et les accumulateurs bruts pour reprendre l'accumulation
- `load(path)` : restaure l'état complet depuis le `.nc`

**Paramètres accumulés** (modifiables dans `config.py`) :

| Variable           | Type | Réduction verticale    | Unité  |
|--------------------|------|------------------------|--------|
| `iwp`              | 1D   | —                      | g/m²   |
| `lwp`              | 1D   | —                      | g/m²   |
| `surface_elevation`| 1D   | —                      | m      |
| `iwc`              | 2D   | nanmean                | g/m³   |
| `lwc`              | 2D   | nanmean                | g/m³   |
| `temperature`      | 2D   | nanmean                | K      |
| `particle_type`    | 2D   | mode dominant (≥1)     | —      |

### `plotting.py`
- Helpers privés : `_add_time_labels`, `_add_particle_legend`,
  `_make_polar_map`, `_add_distance_circles`, `_add_cloud_contours`,
  `_add_secondary_axes`, `_polar_grid_map`
- Colormaps `CMAP_PART` / `CMAP_TEMP` construites une seule fois au chargement

**Figures orbite unique :**

| Fonction | Description |
|---|---|
| `plot_cloud_classification` | Types de particules (pcolormesh) |
| `plot_temperature` | Température en coupe verticale |
| `plot_temperature_and_classification` | Température + contour global du nuage |
| `plot_lat_lon` | Latitude et longitude le long de la trace |
| `plot_distance` | Distance à la station de référence |
| `plot_ice_water_content` | IWC avec axes secondaires UTC et température |
| `plot_liquid_water_content` | LWC avec axes lat/lon supplémentaires |
| `plot_water_paths` | LWP et IWP (profils 1D) |

**Figures cartes polaires :**

| Fonction | Description |
|---|---|
| `plot_polar_scatter` | Scatter coloré sur une orbite |
| `plot_multi_orbit_lwp` | LWP multi-orbites (traces brutes) |
| `plot_grid_lwp` | LWP moyen depuis la grille (remplace `plot_multi_orbit_lwp`) |
| `plot_grid_mean(grid, param)` | Moyenne de n'importe quel paramètre |
| `plot_grid_std(grid, param)` | Écart-type de n'importe quel paramètre |
| `plot_grid_lwp_iwp` | 4 panneaux : LWP mean/std + IWP mean/std |
| `plot_grid_from_nc(filepath, param, stat)` | Trace directement depuis le `.nc` sans GridAccumulator |

### `main.py`
- `run_single_orbit()` : télécharge → traite → 8 figures pour une orbite
- `run_multi_orbit()` : fusionne ESA + JAXA → carte multi-orbites
- `run_grid(force_rebuild=False)` : charge le cache `.nc` si existant,
  sinon reconstruit depuis toutes les orbites avec sauvegarde intermédiaire
  toutes les 10 orbites
- `plot_grid_results(grid)` : enchaîne les 5 cartes statistiques

---

## Cache de la grille

Le fichier `./data/grid_cache.nc` est créé automatiquement au premier
`run_grid()`. Les exécutions suivantes le rechargent instantanément.

```python
# Utilisation directe sans passer par GridAccumulator
import xarray as xr
ds = xr.open_dataset("./data/grid_cache.nc")
ds                          # aperçu interactif dans Jupyter
ds["lwp_mean"]              # moyenne LWP  (n_lat × n_lon)
ds["iwp_std"]               # écart-type IWP
ds.sel(lat=slice(-80, -60)) # sélection géographique

# Tracer sans rien reconstruire
from plotting import plot_grid_from_nc
plot_grid_from_nc("./data/grid_cache.nc", param="lwp", stat="mean")
plot_grid_from_nc("./data/grid_cache.nc", param="iwp", stat="std")

# Forcer le recalcul complet (nouvelles orbites ajoutées)
from main import run_grid
grid = run_grid(force_rebuild=True)
```

---

## Données JAXA

Placer les fichiers `.h5` téléchargés depuis le portail JAXA dans :
```
./data/jaxa/
```
La structure HDF5 étant identique aux fichiers ESA, aucune configuration
supplémentaire n'est nécessaire. `run_multi_orbit()` et `run_grid()` les
intègrent automatiquement.
