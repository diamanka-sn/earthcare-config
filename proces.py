# ============================================================
# processing.py
# Calculs scientifiques : géodésie, masques, temps solaire,
# et préparation des tableaux pour les figures.
# ============================================================

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from config import LAT_REF, LON_REF


# ----------------------------------------------------------------
# Géodésie
# ----------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Distance sphérique (km) entre deux points ou tableaux de points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def distance_to_ref(lat, lon):
    """Distance (km) entre chaque point de la trace et la station de référence."""
    return haversine(lat, lon, LAT_REF, LON_REF)


# ----------------------------------------------------------------
# Temps
# ----------------------------------------------------------------

def calculate_local_times(t0_utc, time_s, lon_arr):
    """Calcule le temps solaire local pour chaque point de la trace.

    Le décalage solaire est estimé à raison de 4 min (240 s) par degré de longitude.

    Parameters
    ----------
    t0_utc : datetime
    time_s : array-like  — secondes depuis t0_utc
    lon_arr : array-like — longitudes correspondantes

    Returns
    -------
    list of datetime
    """
    return [
        t0_utc + timedelta(seconds=float(s) + lon * 240)
        for s, lon in zip(time_s, lon_arr)
    ]


# ----------------------------------------------------------------
# Masques et nettoyage
# ----------------------------------------------------------------

def mask_negative(arr):
    """Retourne une copie avec les valeurs négatives remplacées par NaN."""
    return np.where(arr < 0, np.nan, arr)


def mask_particle_type(particle_type):
    """Retourne deux masques :
    - ``data_masked`` : valeurs hors [0, 13] masquées
    - ``data_plot``   : idem + valeur 0 (clear) masquée pour l'affichage
    """
    data_masked = np.ma.masked_where(
        (particle_type < 0) | (particle_type > 13), particle_type
    )
    data_plot = np.ma.masked_where(data_masked == 0, data_masked)
    return data_masked, data_plot


# ----------------------------------------------------------------
# Préparation des grilles 2D
# ----------------------------------------------------------------

def build_time_grid(time_s, height):
    """Construit la grille temporelle 2D T2D (broadcast de t sur l'axe altitude).

    Parameters
    ----------
    time_s : 1D array — temps absolu en secondes
    height : 2D array — (n_time, n_height)

    Returns
    -------
    t : 1D array  — temps relatif (s depuis début)
    T2D : 2D array — grille temps
    HGT : 2D array — alias height
    """
    t   = time_s - time_s[0]
    T2D = np.tile(t[:, np.newaxis], (1, height.shape[1]))
    return t, T2D, height


# ----------------------------------------------------------------
# Préparation complète d'une orbite unique
# ----------------------------------------------------------------

def prepare_single_orbit(orbit_data: dict, t0_utc) -> dict:
    """Calcule toutes les variables dérivées nécessaires aux figures.

    Parameters
    ----------
    orbit_data : dict  — sortie de ``io_data.load_orbit``
    t0_utc : datetime

    Returns
    -------
    dict avec les clés : t, T2D, HGT, temp_c, data_plot, iwc_plot,
                         lwc_plot, lwp_plot, iwp_plot, distance,
                         local_times, t_utc_start, t_utc_end,
                         present_type
                         + toutes les clés d'orbit_data
    """
    lat               = orbit_data["lat"]
    lon               = orbit_data["lon"]
    height            = orbit_data["height"]
    time_s            = orbit_data["time"]
    temperature       = orbit_data["temperature"]
    particle_type     = orbit_data["particle_type"]
    iwc               = orbit_data["iwc"]
    lwc               = orbit_data["lwc"]
    iwp               = orbit_data["iwp"]
    lwp               = orbit_data["lwp"]

    t, T2D, HGT = build_time_grid(time_s, height)
    _, data_plot = mask_particle_type(particle_type)

    local_times = calculate_local_times(t0_utc, t, lon)
    t_utc_start = (t0_utc + timedelta(seconds=float(t[0]))).strftime("%H:%M UTC")
    t_utc_end   = (t0_utc + timedelta(seconds=float(t[-1]))).strftime("%H:%M UTC")

    present_type = sorted(
        [int(v) for v in np.unique(particle_type) if 1 <= int(v) <= 13]
    )

    return {
        **orbit_data,
        "t":            t,
        "T2D":          T2D,
        "HGT":          HGT,
        "temp_c":       temperature - 273.15,
        "data_plot":    data_plot,
        "iwc_plot":     np.ma.masked_where(iwc < 0, iwc),
        "lwc_plot":     np.ma.masked_where(lwc < 0, lwc),
        "lwp_plot":     mask_negative(lwp),
        "iwp_plot":     mask_negative(iwp),
        "distance":     distance_to_ref(lat, lon),
        "local_times":  local_times,
        "t_utc_start":  t_utc_start,
        "t_utc_end":    t_utc_end,
        "present_type": present_type,
    }


# ----------------------------------------------------------------
# Préparation multi-orbites
# ----------------------------------------------------------------

def _orbit_id(data: dict) -> str:
    """Construit un identifiant unique pour une orbite.

    Format : ORBITE_YYYYMMDD_HHMMSS
    Combine le numero d orbite, la date et l heure UTC de debut
    pour differencier deux passages du meme numero a des jours differents.

    Exemples :
        09039G_20251230_215005
        09040G_20251231_003210
    """
    # Numero d orbite
    num = data.get("nom_orbite", b"unknown")
    if isinstance(num, (bytes, np.bytes_)):
        num = num.decode()
    else:
        num = str(num).strip()

    # Frame ID
    fid = data.get("frame_id", b"G")
    if isinstance(fid, (bytes, np.bytes_)):
        fid = fid.decode()
    else:
        fid = str(fid).strip()
        if fid.startswith("b'") and fid.endswith("'"):
            fid = fid[2:-1]

    # t0_utc : heure UTC de debut du passage
    t0 = data.get("t0_utc")
    if t0 is not None:
        date_str = t0.strftime("%Y%m%d_%H%M%S")
    else:
        date_str = "unknowndate"

    return f"{num}{fid}_{date_str}"


def prepare_multi_orbits(raw_orbits: list[dict]) -> list[dict]:
    """Nettoie la liste brute des orbites pour l affichage multi-traces.

    Retourne une liste de dicts avec les cles :
    orbit_id, nom_orbite, date_debut, lat, lon, lwp, iwp.

    orbit_id est unique meme si deux orbites ont le meme numero
    (passages differents de la meme orbite a des dates differentes).
    Format : NUMEROG_YYYYMMDD_HHMMSS  ex. 09039G_20251230_215005
    """
    result = []
    for data in raw_orbits:
        t0 = data.get("t0_utc")
        result.append({
            "orbit_id":   _orbit_id(data),
            "nom_orbite": str(data.get("nom_orbite", b"unknown")).strip(),
            "date_debut": t0.strftime("%Y-%m-%d %H:%M:%S") if t0 else "unknown",
            "t0_utc":     t0,
            "lat":        np.asarray(data["lat"], dtype=float),
            "lon":        np.asarray(data["lon"], dtype=float),
            "lwp":        mask_negative(data["lwp"]),
            "iwp":        mask_negative(data["iwp"]),
        })
    return result


def save_multi_orbits(orbites: list[dict], filepath: str) -> None:
    """Sauvegarde la liste d orbites preparees dans un fichier NetCDF4.

    Chaque orbite est identifiee par son orbit_id unique
    (numero + date + heure). Le fichier peut etre recharge avec
    load_multi_orbits_nc() sans relire aucun fichier HDF5.

    Parameters
    ----------
    orbites : list[dict]  sortie de prepare_multi_orbits()
    filepath : str        chemin du .nc a creer
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for orb in orbites:
        n = len(orb["lat"])
        ds = xr.Dataset(
            {
                "lat": ("point", orb["lat"]),
                "lon": ("point", orb["lon"]),
                "lwp": ("point", orb["lwp"]),
                "iwp": ("point", orb["iwp"]),
            },
            coords={"point": np.arange(n)},
            attrs={
                "orbit_id":   orb["orbit_id"],
                "nom_orbite": orb["nom_orbite"],
                "date_debut": orb["date_debut"],
            },
        )
        datasets.append(ds)

    combined = xr.concat(datasets, dim="orbite")
    combined["orbite"] = [orb["orbit_id"] for orb in orbites]

    # Metadonnees globales
    combined.attrs["n_orbites"]      = len(orbites)
    combined.attrs["description"]    = "Orbites EarthCARE ACM_CLP_2B preparees"
    combined.attrs["date_premiere"]  = orbites[0]["date_debut"] if orbites else ""
    combined.attrs["date_derniere"]  = orbites[-1]["date_debut"] if orbites else ""

    # Stocker date_debut de chaque orbite comme variable string
    combined["date_debut"] = ("orbite", [orb["date_debut"] for orb in orbites])
    combined["nom_orbite"] = ("orbite", [orb["nom_orbite"] for orb in orbites])

    combined.to_netcdf(filepath, mode="w")
    size_kb = Path(filepath).stat().st_size / 1e3
    print(f"[orbites] Sauvegarde -> {filepath}")
    print(f"          {len(orbites)} orbites | {size_kb:.0f} KB")
    print(f"          Periode : {combined.attrs['date_premiere']}"
          f" -> {combined.attrs['date_derniere']}")


def load_multi_orbits_nc(filepath: str) -> list[dict]:
    """Charge la liste d orbites depuis un fichier NetCDF4.

    Retourne le meme format que prepare_multi_orbits() — compatible
    avec toutes les fonctions de plotting sans modification.

    Parameters
    ----------
    filepath : str  chemin vers un fichier produit par save_multi_orbits()

    Returns
    -------
    list[dict]  avec les cles : orbit_id, nom_orbite, date_debut,
                                t0_utc, lat, lon, lwp, iwp
    """
    from datetime import datetime
    ds = xr.open_dataset(filepath)

    orbites = []
    for i in range(len(ds["orbite"])):
        orb_ds = ds.isel(orbite=i)

        orbit_id   = str(orb_ds["orbite"].values)
        nom_orbite = str(orb_ds["nom_orbite"].values)
        date_debut = str(orb_ds["date_debut"].values)

        # Reconstruire t0_utc depuis date_debut
        try:
            t0_utc = datetime.strptime(date_debut, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            t0_utc = None

        orbites.append({
            "orbit_id":   orbit_id,
            "nom_orbite": nom_orbite,
            "date_debut": date_debut,
            "t0_utc":     t0_utc,
            "lat":        orb_ds["lat"].values.copy(),
            "lon":        orb_ds["lon"].values.copy(),
            "lwp":        orb_ds["lwp"].values.copy(),
            "iwp":        orb_ds["iwp"].values.copy(),
        })

    ds.close()
    print(f"[orbites] Charge <- {filepath}  ({len(orbites)} orbites)")
    if orbites:
        print(f"          Periode : {orbites[0]['date_debut']}"
              f" -> {orbites[-1]['date_debut']}")
    return orbites


def _decode_frame_id(raw) -> str:
    """Décode un frame_id HDF5 (bytes, np.bytes_ ou str) en chaîne propre.

    Exemples :  b"G"  →  "G"
                "b'G'" →  "G"
                "G"   →  "G"
    """
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode()
    s = str(raw)
    # cas np.bytes_ affiché comme "b'G'"
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def build_orbit_label(orbites: list[dict]) -> str:
    """Construit la chaîne lisible des numéros d'orbite pour les titres.

    Le frame_id est commun à toutes les orbites d'un même produit ;
    il est donc extrait une seule fois depuis la première orbite disponible.

    Exemple de sortie : "09039G, 09040G, 09041G"
    """
    if not orbites:
        return ""
    frame_str = _decode_frame_id(orbites[0].get("frame_id", b"G"))
    # Dédoublonnage tout en conservant l'ordre d'apparition
    seen = set()
    labels = []
    for orb in orbites:
        num = str(orb["nom_orbite"][0])
        if num not in seen:
            seen.add(num)
            labels.append(num + frame_str)
    return ", ".join(labels)


#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
# ============================================================
# main.py
# Point d'entrée — orchestre téléchargement, traitement et
# visualisation pour ECA_JXBA_ACM_CLP_2B.
#
# Usage :
#   python main.py
#   # ou dans Jupyter : %run main.py
# ============================================================

from config import (
    ORBIT_FRAME, DATE_START, DATE_END,
    T_MIN, T_MAX,
    DOWNLOAD_PERIODS,
    HDF5_FIELDS_ORBIT_META,
)
from io_data import (
    download_product, download_multi_period,
    search_product, load_orbit, load_multi_orbits, get_t0_utc,
)
from io_jaxa import load_jaxa_orbits, merge_orbit_sources
from gridding import GridAccumulator
from processing import (
    prepare_single_orbit, prepare_multi_orbits, build_orbit_label,
    save_multi_orbits, load_multi_orbits_nc,
)
from plotting import (
    plot_cloud_classification,
    plot_temperature,
    plot_temperature_and_classification,
    plot_lat_lon,
    plot_distance,
    plot_ice_water_content,
    plot_liquid_water_content,
    plot_water_paths,
    plot_polar_scatter,
    plot_multi_orbit_lwp,
    plot_grid_lwp,
    plot_grid_mean,
    plot_grid_std,
    plot_grid_lwp_iwp,
    plot_orbits_by_period,
    plot_distribution,
    plot_time_series,
    plot_correlation_lwp_iwp,
    plot_latitudinal_profile,
    plot_eda,
    descriptive_stats,
    describe_lwp_iwp,
)


# ============================================================
# SECTION 1 — Orbite unique (analyse fine)
# ============================================================

def run_single_orbit():
    """Pipeline complet pour une orbite unique."""

    # --- Téléchargement & recherche ---
    download_product(ORBIT_FRAME, DATE_START, DATE_END)
    ds = search_product(DATE_START, DATE_END, orbit_and_frame=ORBIT_FRAME)
    display(ds)

    fp      = ds.filepath[0]
    t0_utc  = get_t0_utc(fp)
    raw     = load_orbit(fp)

    # --- Prétraitement ---
    d = prepare_single_orbit(raw, t0_utc)
    d["t0_utc"] = t0_utc   # transmis aux fonctions de labels

    print(f"Fichier   : {fp}")
    print(f"Shapes    — particle_type: {d['particle_type'].shape} "
          f"| temperature: {d['temperature'].shape} "
          f"| height: {d['HGT'].shape}")

    # --- Figures section 1 : coupes verticales ---------------
    plot_cloud_classification(d, T_MIN, T_MAX)
    plot_temperature(d, T_MIN, T_MAX)
    plot_temperature_and_classification(d, T_MIN, T_MAX)

    # --- Figures section 2 : profils 1D ---------------------
    plot_lat_lon(d, T_MIN, T_MAX)
    plot_distance(d)
    plot_ice_water_content(d, T_MIN, T_MAX)
    plot_liquid_water_content(d, T_MIN, T_MAX)
    plot_water_paths(d, T_MIN, T_MAX)

    # --- Figures section 3 : cartes polaires ----------------
    plot_polar_scatter(
        d["lon"], d["lat"], d["iwp_plot"],
        title="Liquid Water Path", cbar_label="LWP ($g/m²$)",
        t_utc_start=d["t_utc_start"], t_utc_end=d["t_utc_end"],
        orbit_id=ORBIT_FRAME, vmin=0, vmax=50,
    )
    plot_polar_scatter(
        d["lon"], d["lat"], d["iwp_plot"],
        title="Ice Water Path", cbar_label="IWP ($g/m²$)",
        t_utc_start=d["t_utc_start"], t_utc_end=d["t_utc_end"],
        orbit_id=ORBIT_FRAME, vmin=0, vmax=100,
        gridlines_labels=True,
    )


# ============================================================
# SECTION 2 — Multi-orbites (carte de synthèse ESA + JAXA)
# ============================================================

# Dossier contenant les fichiers .h5 téléchargés depuis le portail JAXA.
# Détecte si on est dans un script ou dans Jupyter.
# __file__ n'existe pas dans Jupyter → répertoire courant utilisé à la place.
try:
    _HERE = Path(__file__).resolve().parent   # python main.py
except NameError:
    _HERE = Path.cwd()                        # Jupyter (%run / notebook)

JAXA_DATA_DIR = str(_HERE / "data" / "jaxa")


def run_multi_orbit(force_rebuild: bool = False):
    """Pipeline multi-orbites sur une saison — fusionne ESA et JAXA.

    Si le cache ORBITES_CACHE existe et que force_rebuild=False,
    les orbites sont rechargees instantanement depuis le .nc
    sans relire aucun fichier HDF5.

    Parameters
    ----------
    force_rebuild : bool, default False
        Forcer le rechargement complet depuis les HDF5.
    """
    from pathlib import Path

    # --- Retour rapide si cache valide ----------------------
    if not force_rebuild and Path(ORBITES_CACHE).exists():
        print(f"[orbites] Cache trouve ({ORBITES_CACHE})")
        orbites = load_multi_orbits_nc(ORBITES_CACHE)
        n_orbites_label = ", ".join(o["orbit_id"] for o in orbites)
        plot_multi_orbit_lwp(orbites, n_orbites_label)
        plot_orbits_by_period(orbites, param="lwp", period_days=5, vmin=0, vmax=40)
        plot_orbits_by_period(orbites, param="iwp", period_days=5, vmin=0, vmax=100)
        return

    # --- Téléchargement et chargement ESA + JAXA ------------
    download_multi_period(DOWNLOAD_PERIODS)
    ds_multi = search_product(DATE_START, DATE_END, frame_id="G")
    display(ds_multi)

    esa_raw  = load_multi_orbits(ds_multi.filepath,
                                 extra_fields=HDF5_FIELDS_ORBIT_META)
    jaxa_raw = load_jaxa_orbits(JAXA_DATA_DIR,
                                extra_fields=HDF5_FIELDS_ORBIT_META)
    all_raw  = merge_orbit_sources(esa_raw, jaxa_raw, sort_by="time")

    orbites         = prepare_multi_orbits(all_raw)
    n_orbites_label = build_orbit_label(all_raw)

    # --- Sauvegarde du cache --------------------------------
    save_multi_orbits(orbites, ORBITES_CACHE)

    # --- Figures --------------------------------------------
    plot_multi_orbit_lwp(orbites, n_orbites_label)
    plot_orbits_by_period(orbites, param="lwp", period_days=5, vmin=0, vmax=40)
    plot_orbits_by_period(orbites, param="iwp", period_days=5, vmin=0, vmax=100)
def run_grid(force_rebuild: bool = False) -> GridAccumulator:
    """Accumule toutes les orbites (ESA + JAXA) sur la grille lat/lon.

    Si le cache ``GRID_CACHE`` existe et que ``force_rebuild=False``,
    la grille est rechargée depuis le disque instantanément sans relire
    aucun fichier HDF5.

    Parameters
    ----------
    force_rebuild : bool, default False
        Forcer le recalcul complet même si le cache existe.

    Returns
    -------
    GridAccumulator  prêt pour grid.mean(), grid.std() ou grid.count()
    """
    cache_path = Path(GRID_CACHE)

    # --- Retour rapide si cache valide --------------------------
    if not force_rebuild and cache_path.exists():
        print(f"[grid] Cache trouvé ({cache_path}) — chargement sans relire les HDF5")
        return GridAccumulator.load(str(cache_path))

    if force_rebuild:
        print("[grid] force_rebuild=True — recalcul complet demandé")
    else:
        print(f"[grid] Cache absent ({cache_path}) — construction depuis les orbites brutes")

    # --- Créer le dossier de cache si nécessaire ----------------
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Chargement ESA + JAXA ----------------------------------
    download_multi_period(DOWNLOAD_PERIODS)
    ds_multi = search_product(DATE_START, DATE_END, frame_id="G")
    esa_raw  = load_multi_orbits(ds_multi.filepath)
    jaxa_raw = load_jaxa_orbits(JAXA_DATA_DIR)
    all_raw  = merge_orbit_sources(esa_raw, jaxa_raw, sort_by="time")

    if not all_raw:
        raise RuntimeError("[grid] Aucune orbite chargée — vérifiez les chemins et dates.")

    # --- Accumulation orbite par orbite -------------------------
    grid = GridAccumulator(dlat=1.0, dlon=10.0)
    for i, orb in enumerate(all_raw, 1):
        grid.accumulate(orb)
        # Sauvegarde intermédiaire toutes les 10 orbites
        # → reprise possible si interruption
        if i % 10 == 0 or i == len(all_raw):
            print(f"  {i}/{len(all_raw)} orbites accumulées...")
            try:
                grid.save(str(cache_path))
            except Exception as e:
                print(f"  [grid] Impossible de sauvegarder le cache : {e}")

    print(f"[grid] Terminé — {grid}")
    print(f"[grid] Cache sauvegardé -> {cache_path.resolve()}")
    return grid

def plot_grid_results(grid) -> None:
    """Trace les cartes de moyenne et d'écart-type depuis la grille.

    À appeler après run_grid() :
        grid = run_grid()
        plot_grid_results(grid)
    """
    plot_grid_lwp(grid)                        # LWP moyen (remplace plot_multi_orbit_lwp)
    plot_grid_mean(grid, "iwp", "IWP ($g/m²$)", vmin=0, vmax=100)
    plot_grid_std(grid,  "lwp", "σ LWP ($g/m²$)")
    plot_grid_std(grid,  "iwp", "σ IWP ($g/m²$)")
    plot_grid_lwp_iwp(grid)                    # vue 4 panneaux : mean + std


# ============================================================
# SECTION 4 — Analyse exploratoire
# ============================================================

def run_eda():
    """Lance toutes les analyses exploratoires sur les orbites brutes.

    Produit : histogrammes, series temporelles,
    correlation LWP/IWP et profils latitudinaux.
    """
    ds_multi = search_product(DATE_START, DATE_END, frame_id="G")
    esa_raw  = load_multi_orbits(ds_multi.filepath)
    jaxa_raw = load_jaxa_orbits(JAXA_DATA_DIR)
    all_raw  = merge_orbit_sources(esa_raw, jaxa_raw, sort_by="time")

    if not all_raw:
        print("[eda] Aucune orbite disponible.")
        return

    print(f"[eda] {len(all_raw)} orbites chargees.")
    describe_lwp_iwp(all_raw)
    plot_eda(all_raw, period_days=5)

# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_single_orbit()
    run_multi_orbit()
    grid = run_grid()
    plot_grid_results(grid)
    run_eda()