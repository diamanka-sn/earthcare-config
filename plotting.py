# ============================================================
# plotting.py
# Fonctions de visualisation pour ECA_JXBA_ACM_CLP_2B.
# Chaque fonction publique (plot_*) produit et affiche une figure.
# Les helpers privés (_add_*, make_*) construisent des éléments
# réutilisables sans déclencher d'affichage.
# ============================================================

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

from config import (
    TITLE_COLOR, TITLE_WEIGHT,
    PARTICLE_COLORS, PARTICLE_LABEL,
    CIRCLES_CONFIG,
    LON_REF, LAT_REF,
    ORBIT_FRAME,
)


# ============================================================
# COLORMAPS (construites une seule fois au chargement du module)
# ============================================================

CMAP_PART  = mcolors.ListedColormap([PARTICLE_COLORS[i] for i in range(14)])
BOUNDS_PART = np.arange(-0.5, 14.5, 1)
NORM_PART  = mcolors.BoundaryNorm(BOUNDS_PART, CMAP_PART.N)

CMAP_TEMP  = plt.cm.rainbow
NORM_TEMP  = mcolors.Normalize(vmin=-48, vmax=-12)


# ============================================================
# HELPERS PRIVÉS
# ============================================================

def _add_time_labels(ax, t, t0_utc, local_times, t_min, t_max,
                     y_utc=-0.18, y_local=-0.25):
    """Ajoute les labels de temps UTC et solaire local sous l'axe."""
    idx_start = np.argmin(np.abs(t - t_min))
    idx_end   = np.argmin(np.abs(t - t_max))
    dt_start  = t0_utc + timedelta(seconds=t_min)
    dt_end    = t0_utc + timedelta(seconds=t_max)
    lst_start = local_times[idx_start]
    lst_end   = local_times[idx_end]

    kw = dict(transform=ax.transAxes, fontsize=10,
              color=TITLE_COLOR, fontweight=TITLE_WEIGHT)

    for y, left_dt, right_dt, center_label in [
        (y_utc,   dt_start,  dt_end,  "Time / UTC"),
        (y_local, lst_start, lst_end, "Time / Local"),
    ]:
        ax.text(0.0, y, left_dt.strftime("%H:%M:%S"),  ha="left",   **kw)
        ax.text(1.0, y, right_dt.strftime("%H:%M:%S"), ha="right",  **kw)
        ax.text(0.5, y, center_label,                  ha="center", **kw)


def _add_particle_legend(ax_leg, present_types):
    """Dessine la légende des types de particules dans un axe dédié."""
    patches = [
        mpatches.Patch(
            facecolor=PARTICLE_COLORS[i], edgecolor="grey",
            linewidth=0.5, label=PARTICLE_LABEL[i],
        )
        for i in present_types
    ]
    ax_leg.axis("off")
    ax_leg.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=min(5, len(present_types)),
        fontsize=10,
        frameon=True,
        edgecolor="grey",
        title="cloud particle type",
        title_fontsize=9,
        borderaxespad=4,
    )


def _make_polar_map(header_text):
    """Crée une figure polaire SouthPolarStereo prête à l'emploi.

    Returns (fig, ax)  — extent, features et boundary circulaire déjà configurés.
    """
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.98, header_text, fontsize=7, va="top")

    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    # Boundary circulaire
    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)

    return fig, ax


def _add_distance_circles(ax):
    """Trace les cercles de distance autour de la station de référence."""
    gd = cgeo.Geodesic()
    for cfg in CIRCLES_CONFIG:
        cp = gd.circle(
            lon=LON_REF, lat=LAT_REF,
            radius=cfg["radius"] * 1000,
            n_samples=100, endpoint=True,
        )
        ax.plot(
            cp[:, 0], cp[:, 1],
            color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=2, transform=ccrs.PlateCarree(),
            zorder=10, label=f"Radius: {cfg['label']}",
        )


def _fig_header(t_utc_start, t_utc_end, orbit_id=None):
    """Construit la chaîne d'en-tête standard des figures."""
    orbit_id = orbit_id or ORBIT_FRAME
    return (
        f"ECA_JXBA_ACM_CLP_2B_20251230T215005Z_20251230T234304Z_{orbit_id}.h5\n"
        f"From : {t_utc_start} to {t_utc_end}\norbit: {orbit_id}"
    )


# ============================================================
# FIGURES VERTICALES (coupes 2D altitude × temps)
# ============================================================

def plot_cloud_classification(d: dict, t_min: int, t_max: int) -> None:
    """Figure 1 — Classification des types de particules."""
    fig, (ax1, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.15},
    )
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.98, _fig_header(d["t_utc_start"], d["t_utc_end"]),
             fontsize=7, va="top")

    ax1.pcolormesh(d["T2D"], d["HGT"], d["data_plot"],
                   cmap=CMAP_PART, norm=NORM_PART, shading="auto")
    ax1.plot(d["t"], d["surface_elevation"],
             color="saddlebrown", linewidth=1.5, label="surface elevation")
    ax1.set_ylabel("Altitude / m", fontsize=10)
    ax1.set_xlabel("time / s", fontsize=9)
    ax1.set_title("Cloud classification", fontsize=13,
                  color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.7)

    _add_time_labels(ax1, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.15, y_local=-0.23)
    _add_particle_legend(ax_leg, d["present_type"])
    plt.show()


def plot_temperature(d: dict, t_min: int, t_max: int) -> None:
    """Figure 2 — Température (coupe altitude × temps)."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")

    c = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                      cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto")
    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95)
    cb.set_label("Temperature en °C", fontsize=8)
    ax.set_ylabel("Altitude / m", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=9)
    ax.set_title("Temperature", fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)
    plt.show()


def plot_temperature_and_classification(d: dict, t_min: int, t_max: int) -> None:
    """Figure 3 — Température et classification superposées."""
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.154},
    )
    fig.patch.set_facecolor("white")

    c = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                      cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto", alpha=1.0)
    ax.pcolormesh(d["T2D"], d["HGT"], d["data_plot"],
                  cmap=CMAP_PART, norm=NORM_PART, shading="auto", alpha=0.45)

    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95,
                      ticks=np.arange(-48, -11, 4))
    cb.set_label("Temperature °C", fontsize=8)
    ax.set_ylabel("Altitude / m", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=9)
    ax.set_title("Temperature and cloud classification",
                 fontsize=12, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)

    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.25)
    _add_particle_legend(ax_leg, d["present_type"])
    plt.show()


def _plot_water_content_2d(d: dict, data_key: str, title: str, cbar_label: str,
                            t_min: int, t_max: int,
                            vmin=None, vmax=None, extra_twin=None) -> None:
    """Générique pcolormesh pour IWC ou LWC (usage interne)."""
    fig, (ax1, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.15},
    )
    fig.patch.set_facecolor("white")

    kwargs = dict(cmap="rainbow", shading="auto")
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax

    c1 = ax1.pcolormesh(d["T2D"], d["HGT"], d[data_key], **kwargs)
    ax1.plot(d["t"], d["surface_elevation"],
             color="saddlebrown", linewidth=1.5, label="surface elevation", zorder=5)
    ax1.set_ylabel("Altitude / m", fontsize=10)
    ax1.set_xlabel("time / s", fontsize=9)
    ax1.set_title(title, fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)

    ax_dist = ax1.twinx()
    ax_dist.plot(d["t"], d["distance"], color="#FFA500",
                 linewidth=1.5, label="Distance to DC")
    ax_dist.set_ylabel("Distance to DC / km", fontsize=10, color="#FFA500")
    ax_dist.tick_params(axis="y", labelcolor="#FFA500")
    ax_dist.set_ylim(600, 1400)
    ax_dist.set_xlim(t_min + 10, t_max)

    if extra_twin:
        extra_twin(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax_dist.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax_leg.axis("off")
    cb = fig.colorbar(c1, ax=ax_leg, orientation="horizontal", fraction=0.5, pad=0.1)
    cb.set_label(cbar_label, fontsize=9)
    plt.show()


def plot_ice_water_content(d: dict, t_min: int, t_max: int) -> None:
    """Figure 5 — Ice Water Content."""
    _plot_water_content_2d(d, "iwc_plot", "Ice water content", "($g/m³$)",
                            t_min, t_max, vmin=0.0, vmax=0.20)


def plot_liquid_water_content(d: dict, t_min: int, t_max: int) -> None:
    """Figure 6 — Liquid Water Content (avec axes lat/lon secondaires)."""
    def _add_latlon(ax1):
        ax_lat = ax1.twinx()
        ax_lat.plot(d["t"], d["lat"], color="#FFA500", linewidth=1.5, label="Latitude")
        ax_lat.set_ylabel("Latitude / deg", fontsize=10, color="#FFA500",
                          fontweight=TITLE_WEIGHT)
        ax_lat.tick_params(axis="y", labelcolor="#FFA500")

        ax_lon = ax1.twinx()
        ax_lon.spines["right"].set_position(("axes", 1.06))
        ax_lon.plot(d["t"], d["lon"], color="#E65100", linewidth=1.5, label="Longitude")
        ax_lon.set_ylabel("Longitude / deg", fontsize=10, color="#E65100",
                          fontweight=TITLE_WEIGHT)
        ax_lon.tick_params(axis="y", labelcolor="#E65100")
        ax_lon.set_ylim(50, 140)

    _plot_water_content_2d(d, "lwc_plot", "Liquid water content", "($g/m³$)",
                            t_min, t_max, extra_twin=_add_latlon)


# ============================================================
# FIGURES 1D (profils temporels)
# ============================================================

def plot_lat_lon(d: dict, t_min: int, t_max: int) -> None:
    """Figure 4 — Latitude et longitude le long de la trace."""
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("white")
    ax_r = ax.twinx()

    ax.plot(d["t"], d["lon"], color=TITLE_COLOR, linewidth=1.5, label="Longitude")
    ax.axhline(y=LON_REF, color=TITLE_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax_r.plot(d["t"], d["lat"], color="#FFA500", linewidth=1.5, label="Latitude")
    ax_r.axhline(y=LAT_REF, color="#FFA500", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_ylabel("Longitude / deg", fontsize=9, color=TITLE_COLOR)
    ax_r.set_ylabel("Latitude / deg", fontsize=9, color="#FFA500")
    ax_r.tick_params(axis="y", labelcolor="#FFA500")
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(50, 140)
    ax.set_xlabel("Time / s", fontsize=10)
    ax.set_title("Latitude et longitude", fontsize=13,
                 color=TITLE_COLOR, fontweight=TITLE_WEIGHT)

    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.30)
    plt.show()


def plot_distance(d: dict) -> None:
    """Figure 5 — Distance à la station de référence."""
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("white")
    ax.plot(d["t"], d["distance"], color=TITLE_COLOR, linewidth=1.5)
    ax.set_ylabel("Distance / km", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=10)
    ax.set_title("Distance to Concordia", fontsize=13,
                 color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_xlim(460, 580)
    ax.set_ylim(700, 1200)
    plt.show()


def plot_water_paths(d: dict, t_min: int, t_max: int) -> None:
    """Figure 8 — LWP et IWP (profils 1D)."""
    fig, ax_lwp = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")

    line1, = ax_lwp.plot(d["t"], d["lwp_plot"], color="red",
                          linewidth=1.5, label="Liquid Water Path (LWP)")
    ax_lwp.set_ylabel("LWP / $g/m²$", color="red", fontsize=10, fontweight=TITLE_WEIGHT)
    ax_lwp.tick_params(axis="y", labelcolor="red")
    ax_lwp.set_ylim(0, 80)
    ax_lwp.set_xlim(t_min, t_max)

    ax_iwp = ax_lwp.twinx()
    line2, = ax_iwp.plot(d["t"], d["iwp_plot"], color="blue",
                          linewidth=1.5, label="Ice Water Path (IWP)")
    ax_iwp.set_ylabel("IWP / $g/m²$", fontsize=10, color="blue", fontweight=TITLE_WEIGHT)
    ax_iwp.tick_params(axis="y", labelcolor="blue")
    ax_iwp.set_ylim(0, 120)
    ax_iwp.set_xlim(t_min, t_max)

    ax_lwp.set_xlabel("Time / s", fontsize=9)
    ax_lwp.legend([line1, line2], [line1.get_label(), line2.get_label()],
                   loc="upper right", fontsize=8)
    plt.show()


# ============================================================
# FIGURES CARTES POLAIRES
# ============================================================

def plot_polar_scatter(lon_arr, lat_arr, data_scatter, title, cbar_label,
                       t_utc_start, t_utc_end, orbit_id=None,
                       vmin=0, vmax=50, gridlines_labels=False,
                       cmap="rainbow") -> None:
    """Carte polaire SouthPolarStereo avec scatter coloré (orbite unique)."""
    fig, ax = _make_polar_map(
        _fig_header(t_utc_start, t_utc_end, orbit_id) + "\n"
    )

    if gridlines_labels:
        ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                     y_inline=True, alpha=0.5)
    else:
        ax.gridlines()

    _add_distance_circles(ax)

    sc = ax.scatter(lon_arr, lat_arr, c=data_scatter, cmap=cmap, s=10,
                    transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_multi_orbit_lwp(orbites: list[dict], n_orbites_label: str) -> None:
    """Carte polaire LWP pour un ensemble d'orbites."""
    fig, ax = _make_polar_map(
        f"ECA_JXBA_ACM_CLP_2B\nOrbits: {n_orbites_label}\nDate: 30/12/2025"
    )

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.3,
                      ylocs=np.arange(-90, -55, 10))
    gl.ylabel_style = {"size": 10}
    gl.xlabel_style = {"size": 10}
    gl.top_labels   = False
    gl.right_label  = False

    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", linestyle="none",
            markersize=5, label="Dome C", transform=ccrs.PlateCarree())

    sc_lwp = None
    for orb in orbites:
        sc_lwp = ax.scatter(orb["lon"], orb["lat"], c=orb["lwp"],
                             cmap="rainbow", s=5,
                             transform=ccrs.PlateCarree(), vmin=0, vmax=40)

    _add_distance_circles(ax)

    if sc_lwp is not None:
        cbar = plt.colorbar(sc_lwp, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_label("LWP ($g/m²$)", fontsize=10)

    ax.set_title("Liquid Water Path", fontsize=10,
                 fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()
