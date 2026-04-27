# ============================================================
# gridding.py
# Accumulation des paramètres d'intérêt sur une grille
# géographique régulière (lat × lon) et persistance sur disque.
#
# Principe
# --------
# Pour chaque cellule (lat_i, lon_j) de la grille on stocke :
#   - la somme des valeurs  (sum_*)
#   - le nombre de mesures  (count_*)
# La moyenne par cellule = sum / count.
# On sauvegarde l'état intermédiaire en .npz après chaque lot
# d'orbites, ce qui permet de reprendre sans tout recharger.
#
# Usage typique
# -------------
#   grid = GridAccumulator()           # ou GridAccumulator.load("grid.npz")
#   for orbit_data in all_orbits:
#       grid.accumulate(orbit_data)
#   grid.save("grid.npz")
#   means = grid.mean()                # dict {param: ndarray (n_lat, n_lon)}
# ============================================================

from pathlib import Path

import numpy as np

from config import HDF5_FIELDS


# ============================================================
# CONFIGURATION DE LA GRILLE
# ============================================================

# Résolution par défaut (modifiable à l'instanciation)
DEFAULT_DLAT =  1.0   # °
DEFAULT_DLON = 10.0   # °

# Paramètres scalaires (intégrés sur la colonne) à accumuler.
# Ces variables sont issues du dict retourné par load_orbit() /
# prepare_multi_orbits() et ont toutes la même dimension (n_temps,).
GRID_PARAMS_1D = [
    "iwp",           # Ice Water Path   (g/m²)
    "lwp",           # Liquid Water Path (g/m²)
    "surface_elevation",
]

# Paramètres 2D (n_temps × n_height) : on stocke la moyenne verticale
# sur tous les niveaux non-NaN de chaque profil avant d'accumuler.
GRID_PARAMS_2D = [
    "iwc",           # Ice Water Content   (g/m³)
    "lwc",           # Liquid Water Content (g/m³)
    "temperature",   # Température         (K, convertie en °C à la demande)
    "particle_type", # Type de particule dominant
]


# ============================================================
# CLASSE PRINCIPALE
# ============================================================

class GridAccumulator:
    """Grille lat × lon qui accumule des orbites EarthCARE.

    Parameters
    ----------
    dlat : float, default 1.0
        Pas de latitude en degrés.
    dlon : float, default 10.0
        Pas de longitude en degrés.
    lat_range : tuple, default (-90, 90)
        Bornes de latitude (incluses).
    lon_range : tuple, default (-180, 180)
        Bornes de longitude (incluses).
    """

    def __init__(self,
                 dlat: float = DEFAULT_DLAT,
                 dlon: float = DEFAULT_DLON,
                 lat_range: tuple = (-90.0, 90.0),
                 lon_range: tuple = (-180.0, 180.0)):

        self.dlat = dlat
        self.dlon = dlon

        # Centres des cellules
        self.lat_bins = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
        self.lon_bins = np.arange(lon_range[0], lon_range[1] + dlon, dlon)
        self.n_lat = len(self.lat_bins)
        self.n_lon = len(self.lon_bins)

        shape = (self.n_lat, self.n_lon)
        all_params = GRID_PARAMS_1D + GRID_PARAMS_2D

        # Tableaux d'accumulation
        self._sum   = {p: np.zeros(shape) for p in all_params}
        self._count = {p: np.zeros(shape, dtype=np.int32) for p in all_params}

        # Nombre d'orbites traitées (pour info)
        self.n_orbits = 0

    # ----------------------------------------------------------------
    # Indexation géographique
    # ----------------------------------------------------------------

    def _cell_indices(self, lat_arr: np.ndarray, lon_arr: np.ndarray):
        """Retourne les indices (i_lat, i_lon) pour chaque point de la trace.

        Les points hors grille sont signalés par un index -1.
        """
        i_lat = np.floor((lat_arr - self.lat_bins[0]) / self.dlat).astype(int)
        i_lon = np.floor((lon_arr - self.lon_bins[0]) / self.dlon).astype(int)

        # Clamp : points exactement sur le bord supérieur → dernière cellule
        i_lat = np.clip(i_lat, 0, self.n_lat - 1)
        i_lon = np.clip(i_lon, 0, self.n_lon - 1)

        return i_lat, i_lon

    # ----------------------------------------------------------------
    # Réduction 2D → scalaire par profil
    # ----------------------------------------------------------------

    @staticmethod
    def _column_mean(arr2d: np.ndarray) -> np.ndarray:
        """Moyenne verticale ignorant les NaN — retourne un vecteur (n_temps,)."""
        with np.errstate(all="ignore"):
            return np.nanmean(arr2d, axis=1)

    @staticmethod
    def _dominant_particle(arr2d: np.ndarray) -> np.ndarray:
        """Type de particule le plus fréquent (hors 0=clear) par profil."""
        n_time = arr2d.shape[0]
        result = np.zeros(n_time, dtype=float)
        for i in range(n_time):
            col = arr2d[i, :]
            valid = col[(col >= 1) & (col <= 13)]
            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                result[i] = vals[np.argmax(counts)]
        return result

    # ----------------------------------------------------------------
    # Accumulation
    # ----------------------------------------------------------------

    def accumulate(self, orbit_data: dict) -> None:
        """Ajoute une orbite à la grille.

        Parameters
        ----------
        orbit_data : dict
            Dict retourné par ``load_orbit()`` ou ``load_jaxa_orbits()``.
            Doit contenir au minimum les clés ``lat``, ``lon``, et les
            paramètres listés dans GRID_PARAMS_1D / GRID_PARAMS_2D.
        """
        lat = orbit_data["lat"]
        lon = orbit_data["lon"]
        i_lat, i_lon = self._cell_indices(lat, lon)

        # --- Paramètres 1D -----------------------------------------
        for param in GRID_PARAMS_1D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            values = np.where(raw < 0, np.nan, raw).astype(float)
            _accumulate_1d(self._sum[param], self._count[param],
                           i_lat, i_lon, values)

        # --- Paramètres 2D (réduction verticale avant accumulation) --
        for param in GRID_PARAMS_2D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            if param == "particle_type":
                values = self._dominant_particle(raw.astype(float))
            else:
                arr = np.where(raw < 0, np.nan, raw).astype(float)
                values = self._column_mean(arr)

            _accumulate_1d(self._sum[param], self._count[param],
                           i_lat, i_lon, values)

        self.n_orbits += 1

    # ----------------------------------------------------------------
    # Extraction des moyennes
    # ----------------------------------------------------------------

    def mean(self) -> dict:
        """Retourne la moyenne par cellule pour chaque paramètre.

        Returns
        -------
        dict {param: ndarray (n_lat, n_lon)}  — NaN pour les cellules vides
        """
        result = {}
        for param in list(self._sum):
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(
                    self._count[param] > 0,
                    self._sum[param] / self._count[param],
                    np.nan,
                )
            result[param] = m
        return result

    def count(self) -> dict:
        """Retourne le nombre de mesures par cellule."""
        return {p: self._count[p].copy() for p in self._count}

    # ----------------------------------------------------------------
    # Persistance
    # ----------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Sauvegarde l'état courant de la grille dans un fichier .npz.

        Le fichier contient les tableaux sum_* et count_* ainsi que les
        métadonnées de la grille (pas, bornes, nombre d'orbites).
        Il peut être rechargé avec ``GridAccumulator.load(filepath)``.
        """
        arrays = {
            "_meta_dlat":     np.array([self.dlat]),
            "_meta_dlon":     np.array([self.dlon]),
            "_meta_lat_bins": self.lat_bins,
            "_meta_lon_bins": self.lon_bins,
            "_meta_n_orbits": np.array([self.n_orbits]),
        }
        for param, arr in self._sum.items():
            arrays[f"sum_{param}"]   = arr
        for param, arr in self._count.items():
            arrays[f"count_{param}"] = arr

        np.savez_compressed(filepath, **arrays)
        print(f"[grid] Grille sauvegardée → {filepath}  "
              f"({self.n_orbits} orbites, "
              f"{self.n_lat}×{self.n_lon} cellules)")

    @classmethod
    def load(cls, filepath: str) -> "GridAccumulator":
        """Charge une grille depuis un fichier .npz existant.

        Permet de reprendre l'accumulation sans recharger les orbites déjà
        traitées.

        Parameters
        ----------
        filepath : str
            Chemin vers un fichier produit par ``GridAccumulator.save()``.

        Returns
        -------
        GridAccumulator  avec l'état restauré
        """
        data = np.load(filepath, allow_pickle=False)

        dlat     = float(data["_meta_dlat"][0])
        dlon     = float(data["_meta_dlon"][0])
        lat_bins = data["_meta_lat_bins"]
        lon_bins = data["_meta_lon_bins"]

        g = cls.__new__(cls)
        g.dlat     = dlat
        g.dlon     = dlon
        g.lat_bins = lat_bins
        g.lon_bins = lon_bins
        g.n_lat    = len(lat_bins)
        g.n_lon    = len(lon_bins)
        g.n_orbits = int(data["_meta_n_orbits"][0])
        g._sum     = {}
        g._count   = {}

        all_params = GRID_PARAMS_1D + GRID_PARAMS_2D
        for param in all_params:
            g._sum[param]   = data[f"sum_{param}"].copy()
            g._count[param] = data[f"count_{param}"].copy()

        print(f"[grid] Grille chargée ← {filepath}  "
              f"({g.n_orbits} orbites déjà traitées, "
              f"{g.n_lat}×{g.n_lon} cellules)")
        return g

    # ----------------------------------------------------------------
    # Représentation
    # ----------------------------------------------------------------

    def __repr__(self):
        cells_with_data = int(np.any(
            [self._count[p] > 0 for p in self._count], axis=0
        ).sum())
        return (
            f"GridAccumulator("
            f"dlat={self.dlat}°, dlon={self.dlon}°, "
            f"{self.n_lat}×{self.n_lon} cells, "
            f"{self.n_orbits} orbits, "
            f"{cells_with_data} cells with data)"
        )


# ============================================================
# HELPER INTERNE — accumulation vectorisée
# ============================================================

def _accumulate_1d(sum_arr, count_arr, i_lat, i_lon, values):
    """Ajoute ``values`` dans les cellules (i_lat, i_lon) en ignorant les NaN.

    Utilise np.add.at pour gérer correctement les doublons d'indices
    (plusieurs points dans la même cellule lors d'un même appel).
    """
    valid = ~np.isnan(values)
    if not np.any(valid):
        return
    np.add.at(sum_arr,   (i_lat[valid], i_lon[valid]), values[valid])
    np.add.at(count_arr, (i_lat[valid], i_lon[valid]), 1)
