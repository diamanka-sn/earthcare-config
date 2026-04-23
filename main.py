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
# SECTION 2 — Multi-orbites (carte de synthèse)
# ============================================================

def run_multi_orbit():
    """Pipeline pour l'analyse multi-orbites sur une saison."""

    # --- Téléchargement des périodes définies dans config ---
    download_multi_period(DOWNLOAD_PERIODS)

    ds_multi = search_product(DATE_START, DATE_END, frame_id="G")
    display(ds_multi)
    print(ds_multi.filepath[:])

    # --- Chargement de toutes les orbites -------------------
    raw_orbits = load_multi_orbits(ds_multi.filepath,
                                   extra_fields=HDF5_FIELDS_ORBIT_META)

    orbites        = prepare_multi_orbits(raw_orbits)
    n_orbites_label = build_orbit_label(raw_orbits)

    # --- Carte multi-orbites --------------------------------
    plot_multi_orbit_lwp(orbites, n_orbites_label)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_single_orbit()
    run_multi_orbit()
