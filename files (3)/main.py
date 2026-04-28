#!/usr/bin/env python
# coding: utf-8
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
# Modifier ce chemin selon votre arborescence locale.
JAXA_DATA_DIR = "./data/jaxa"


def run_multi_orbit():
    """Pipeline multi-orbites sur une saison — fusionne ESA et JAXA.

    Les fichiers JAXA partagent la même structure HDF5 que les fichiers ESA :
    load_orbit() est réutilisé tel quel, sans adaptation.
    """

    # --- Téléchargement et chargement ESA -------------------
    download_multi_period(DOWNLOAD_PERIODS)
    ds_multi = search_product(DATE_START, DATE_END, frame_id="G")
    display(ds_multi)

    esa_raw  = load_multi_orbits(ds_multi.filepath,
                                 extra_fields=HDF5_FIELDS_ORBIT_META)

    # --- Chargement des orbites JAXA (même HDF5_FIELDS) -----
    jaxa_raw = load_jaxa_orbits(JAXA_DATA_DIR,
                                extra_fields=HDF5_FIELDS_ORBIT_META)
    # --- Fusion chronologique ESA + JAXA --------------------
    all_raw = merge_orbit_sources(esa_raw, jaxa_raw, sort_by="time")

    orbites         = prepare_multi_orbits(all_raw)
    n_orbites_label = build_orbit_label(all_raw)

    # --- Carte multi-orbites --------------------------------
    plot_multi_orbit_lwp(orbites, n_orbites_label)



# ============================================================
# SECTION 3 — Grille géographique (cache .npz)
# ============================================================

# Fichier de cache : l'accumulation reprend où elle s'est arrêtée
# si le fichier existe déjà.
GRID_CACHE = "./data/grid_cache.nc"


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
    from pathlib import Path

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
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_single_orbit()
    run_multi_orbit()
    grid = run_grid()
    plot_grid_results(grid)
