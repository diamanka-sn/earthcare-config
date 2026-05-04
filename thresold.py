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
from matplotlib.lines import Line2D
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


def _add_particle_legend(ax_leg, present_types, extra_handles=None):
    """Dessine la légende des types de particules dans un axe dédié.

    Parameters
    ----------
    extra_handles : list, optional
        Handles matplotlib supplémentaires ajoutés après les patches
        (ex. proxy Line2D pour un contour de nuage).
    """
    patches = [
        mpatches.Patch(
            facecolor=PARTICLE_COLORS[i], edgecolor="grey",
            linewidth=0.5, label=PARTICLE_LABEL[i],
        )
        for i in present_types
    ]
    handles = patches + (extra_handles or [])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=handles,
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


def _add_cloud_contours(ax, T2D, HGT, particle_type_raw,
                        color="white", linewidth=0.8, alpha=0.9):
    """Trace le contour global de la masse nuageuse (nuage vs ciel clair).

    Un unique masque binaire : 1 = nuage (type >= 1), 0 = ciel clair.
    Le contour au niveau 0.5 donne la frontière extérieure de l ensemble
    du nuage, indépendamment des types de particules.

    Parameters
    ----------
    particle_type_raw : 2D array — valeurs entières brutes (non masquées)
    color : str, default "white"
    linewidth : float, default 0.8
    alpha : float, default 0.9

    Returns
    -------
    Line2D  proxy utilisable directement dans ax.legend(handles=[...])
    """
    cloud_mask = np.where(particle_type_raw >= 1, 1.0, 0.0)
    ax.contour(
        T2D, HGT, cloud_mask,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        alpha=alpha,
    )
    return Line2D([0], [0], color=color, linewidth=linewidth,
                  alpha=alpha, label="cloud outline")


def plot_temperature_and_classification(d: dict, t_min: int, t_max: int) -> None:
    """Figure 3 — Température avec contours des types de nuages superposés."""
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.154},
    )
    fig.patch.set_facecolor("white")

    # Fond : température en couleur continue
    c = ax.pcolormesh(d["T2D"], d["HGT"], np.ma.masked_invalid(d["temp_c"]),
                      cmap=CMAP_TEMP, norm=NORM_TEMP, shading="auto")
    cb = plt.colorbar(c, ax=ax, pad=0.01, aspect=25, shrink=0.95,
                      ticks=np.arange(-48, -11, 4))
    cb.set_label("Temperature °C", fontsize=8)

    # Contour global du nuage (présence/absence, indépendant du type)
    contour_proxy = _add_cloud_contours(ax, d["T2D"], d["HGT"], d["particle_type"])

    ax.set_ylabel("Altitude / m", fontsize=10)
    ax.set_xlabel("Time / s", fontsize=9)
    ax.set_title("Temperature and cloud classification",
                 fontsize=12, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(3000, 6000)
    ax.set_xlim(t_min, t_max)

    _add_time_labels(ax, d["t"], d["t0_utc"], d["local_times"],
                     t_min, t_max, y_utc=-0.20, y_local=-0.25)
    _add_particle_legend(ax_leg, d["present_type"], extra_handles=[contour_proxy])
    plt.show()


def _add_secondary_axes(ax1, d, t_min, t_max, alt_min=3000, alt_max=6000):
    """Ajoute un axe X secondaire (UTC) en haut et un axe Y secondaire
    (température °C) à droite de l axe principal.

    Axe X secondaire
    ----------------
    Les ticks sont positionnés sur des minutes rondes dans la fenêtre
    [t_min, t_max]. Chaque tick affiche l heure UTC correspondante.

    Axe Y secondaire (température)
    --------------------------------
    La température moyenne sur la fenêtre temporelle est calculée pour
    chaque niveau d altitude, puis interpolée sur les ticks de l axe Y
    principal afin d afficher °C en face de chaque altitude.
    """
    # --- Axe X secondaire : temps UTC en haut -------------------
    ax_top = ax1.twiny()
    ax_top.set_xlim(t_min, t_max)

    # Ticks sur les minutes rondes comprises dans la fenêtre
    t0 = d["t0_utc"]
    dt_min = t0 + timedelta(seconds=t_min)
    dt_max = t0 + timedelta(seconds=t_max)
    # Première minute ronde >= dt_min
    first = dt_min.replace(second=0, microsecond=0)
    if first < dt_min:
        first += timedelta(minutes=1)

    tick_dts, tick_pos = [], []
    cur = first
    while cur <= dt_max:
        tick_pos.append((cur - t0).total_seconds())
        tick_dts.append(cur)
        cur += timedelta(minutes=1)

    ax_top.set_xticks(tick_pos)
    ax_top.set_xticklabels(
        [dt.strftime("%H:%M") for dt in tick_dts],
        fontsize=7, color=TITLE_COLOR,
    )
    ax_top.set_xlabel("Time / UTC", fontsize=8, color=TITLE_COLOR)
    ax_top.tick_params(axis="x", colors=TITLE_COLOR, length=3)

    # --- Axe Y secondaire : température moyenne en °C -----------
    ax_right = ax1.twinx()

    # Moyenne temporelle sur la fenêtre pour chaque niveau d altitude
    mask_t = (d["t"] >= t_min) & (d["t"] <= t_max)
    temp_profile = np.nanmean(d["temp_c"][mask_t, :], axis=0)   # (n_height,)
    hgt_profile  = np.nanmean(d["HGT"][mask_t, :],    axis=0)   # (n_height,)

    # Interpolation : altitude → température pour les ticks Y principaux
    alt_ticks = ax1.get_yticks()
    alt_ticks = alt_ticks[(alt_ticks >= alt_min) & (alt_ticks <= alt_max)]
    temp_at_ticks = np.interp(alt_ticks, hgt_profile, temp_profile)

    ax_right.set_ylim(alt_min, alt_max)
    # Déplacer l axe distance déjà présent — ax_right devient le 3e axe Y
    ax_right.spines["right"].set_position(("axes", 1.0))
    ax_right.set_yticks(alt_ticks)
    ax_right.set_yticklabels(
        [f"{v:.0f}°C" for v in temp_at_ticks],
        fontsize=7, color="#555555",
    )
    ax_right.set_ylabel("Temperature (mean profile)", fontsize=8, color="#555555")
    ax_right.tick_params(axis="y", colors="#555555", length=3)
    # Pas de spine supplémentaire visible
    ax_right.spines["right"].set_visible(False)

    return ax_top, ax_right


def _plot_water_content_2d(d: dict, data_key: str, title: str, cbar_label: str,
                            t_min: int, t_max: int,
                            vmin=None, vmax=None, extra_twin=None) -> None:
    """Générique pcolormesh pour IWC ou LWC avec axes secondaires UTC et température."""
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
    ax1.set_xlabel("Time / s", fontsize=9)
    ax1.set_title(title, fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax1.set_ylim(3000, 6000)
    ax1.set_xlim(t_min, t_max)

    # Axe Y droit : distance à Dome C
    ax_dist = ax1.twinx()
    ax_dist.plot(d["t"], d["distance"], color="#FFA500",
                 linewidth=1.5, label="Distance to DC")
    ax_dist.set_ylabel("Distance to DC / km", fontsize=10, color="#FFA500")
    ax_dist.tick_params(axis="y", labelcolor="#FFA500")
    ax_dist.set_ylim(600, 1400)
    ax_dist.set_xlim(t_min + 10, t_max)

    # Axes secondaires : UTC en haut, température à droite
    _add_secondary_axes(ax1, d, t_min, t_max)

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

# ============================================================
# FIGURES GRILLE GÉOGRAPHIQUE (GridAccumulator)
# ============================================================

def _polar_grid_map(lon_bins, lat_bins, data_2d, title, cbar_label,
                    vmin=None, vmax=None, cmap="rainbow",
                    n_orbits=None):
    """Helper interne : carte polaire pcolormesh depuis une grille lat×lon.

    Parameters
    ----------
    lon_bins, lat_bins : 1D arrays  — centres des cellules
    data_2d : ndarray (n_lat, n_lon)
    n_orbits : int, optional  — affiché dans le sous-titre si fourni
    """
    LON2D, LAT2D = np.meshgrid(lon_bins, lat_bins)

    subtitle = f"{len(lat_bins)}×{len(lon_bins)} cells"
    if n_orbits is not None:
        subtitle += f"  |  {n_orbits} orbits"

    fig, ax = _make_polar_map(subtitle)
    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=False, alpha=0.3,
                 ylocs=np.arange(-90, -55, 10))

    _add_distance_circles(ax)
    ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=6,
            linestyle="none", label="Dome C", transform=ccrs.PlateCarree())

    pc = ax.pcolormesh(LON2D, LAT2D, data_2d,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto")

    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(loc="upper right")
    plt.show()


def plot_grid_lwp(grid) -> None:
    """Carte polaire LWP moyen depuis une grille GridAccumulator.

    Remplace plot_multi_orbit_lwp en utilisant la grille précalculée
    plutôt que les traces orbit-par-orbit. Le résultat est identique
    mais beaucoup plus rapide (plus de boucle sur les orbites).
    """
    means = grid.mean()
    _polar_grid_map(
        grid.lon_bins, grid.lat_bins,
        means["lwp"],
        title="Liquid Water Path — mean",
        cbar_label="LWP ($g/m²$)",
        vmin=0, vmax=40,
        n_orbits=grid.n_orbits,
    )


def plot_grid_mean(grid, param: str = "lwp",
                   cbar_label: str | None = None,
                   vmin=None, vmax=None) -> None:
    """Carte polaire de la moyenne d'un paramètre quelconque de la grille.

    Parameters
    ----------
    param : str
        Nom du paramètre (ex. "lwp", "iwp", "iwc", "lwc", "temperature").
    cbar_label : str, optional
        Étiquette de la colorbar. Si None, utilise ``param``.
    """
    means = grid.mean()
    if param not in means:
        raise KeyError(f"Paramètre '{param}' absent de la grille. "
                       f"Disponibles : {list(means)}")
    _polar_grid_map(
        grid.lon_bins, grid.lat_bins,
        means[param],
        title=f"{param.upper()} — mean",
        cbar_label=cbar_label or param,
        vmin=vmin, vmax=vmax,
        n_orbits=grid.n_orbits,
    )


def plot_grid_std(grid, param: str = "lwp",
                  cbar_label: str | None = None,
                  vmin=0, vmax=None) -> None:
    """Carte polaire de l'écart-type d'un paramètre de la grille.

    Parameters
    ----------
    param : str
        Nom du paramètre (ex. "lwp", "iwp", "iwc", "lwc", "temperature").
    """
    stds = grid.std()
    if param not in stds:
        raise KeyError(f"Paramètre '{param}' absent de la grille. "
                       f"Disponibles : {list(stds)}")
    _polar_grid_map(
        grid.lon_bins, grid.lat_bins,
        stds[param],
        title=f"{param.upper()} — std dev",
        cbar_label=cbar_label or f"σ {param}",
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        n_orbits=grid.n_orbits,
    )


def plot_grid_lwp_iwp(grid) -> None:
    """Deux cartes côte à côte : LWP moyen et IWP moyen.

    Vue de synthèse rapide pour comparer les deux colonnes d'eau.
    """
    means = grid.mean()
    stds  = grid.std()
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)

    proj = ccrs.SouthPolarStereo()
    fig  = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"LWP & IWP  |  {grid.n_orbits} orbits  |  "
        f"grid {grid.dlat}°×{grid.dlon}°",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    specs = [
        ("lwp", "LWP mean ($g/m²$)",   0,  40,  "rainbow"),
        ("iwp", "IWP mean ($g/m²$)",   0, 100,  "rainbow"),
        ("lwp", "LWP std ($g/m²$)",    0,  20,  "YlOrRd"),
        ("iwp", "IWP std ($g/m²$)",    0,  50,  "YlOrRd"),
    ]
    sources = [means["lwp"], means["iwp"], stds["lwp"], stds["iwp"]]
    labels  = ["LWP mean", "IWP mean", "LWP std", "IWP std"]

    for col, (data, (param, cbar_label, vmin, vmax, cmap), label) in enumerate(
        zip(sources, specs, labels)
    ):
        ax = fig.add_subplot(1, 4, col + 1, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)

        theta  = np.linspace(0, 2 * np.pi, 100)
        verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
                linestyle="none", transform=ccrs.PlateCarree())
        _add_distance_circles(ax)

        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)
        ax.set_title(label, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()

def plot_grid_from_nc(filepath: str, param: str = "lwp",
                      stat: str = "mean",
                      vmin=None, vmax=None) -> None:
    """Charge et trace directement depuis un fichier NetCDF4 sans GridAccumulator.

    C'est la manière la plus simple de visualiser la grille après calcul :
    aucun objet Python à reconstruire, juste un chemin de fichier.

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier .nc produit par GridAccumulator.save().
    param : str
        Paramètre à afficher (ex. "lwp", "iwp", "iwc", "temperature"...).
    stat : str
        Statistique : "mean", "std" ou "count".
    vmin, vmax : float, optional
        Bornes de la colorbar.

    Example
    -------
    >>> plot_grid_from_nc("./data/grid_cache.nc", param="lwp", stat="mean")
    >>> plot_grid_from_nc("./data/grid_cache.nc", param="iwp", stat="std")
    """
    import xarray as xr

    ds    = xr.open_dataset(filepath)
    vname = f"{param}_{stat}"

    if vname not in ds:
        available = [v for v in ds.data_vars if not v.startswith("_")]
        raise KeyError(f"Variable '{vname}' absente. Disponibles : {available}")

    data   = ds[vname].values
    lats   = ds["lat"].values
    lons   = ds["lon"].values
    n_orb  = int(ds.attrs.get("n_orbits", 0))
    units  = ds[vname].attrs.get("units", "")
    lname  = ds[vname].attrs.get("long_name", vname)
    ds.close()

    _polar_grid_map(
        lons, lats, data,
        title=lname,
        cbar_label=f"{param} {stat} ({units})",
        vmin=vmin, vmax=vmax,
        cmap="YlOrRd" if stat == "std" else "rainbow",
        n_orbits=n_orb,
    )


def plot_orbits_by_period(orbites: list[dict],
                          param: str = "lwp",
                          period_days: int = 5,
                          vmin: float = 0,
                          vmax: float = 40,
                          cmap: str = "rainbow") -> None:
    """Trace une carte polaire par periode de N jours.

    Les orbites sont regroupees par tranches de ``period_days`` jours
    en se basant sur le temps absolu de chaque orbite. Une figure
    separee est produite pour chaque periode.

    Parameters
    ----------
    orbites : list[dict]
        Liste des orbites preparees (sortie de prepare_multi_orbits).
        Chaque dict doit contenir les cles lon, lat, time
        et le parametre demande (lwp ou iwp).
    param : str, default "lwp"
        Parametre a afficher : "lwp" ou "iwp".
    period_days : int, default 5
        Nombre de jours par periode.
    vmin, vmax : float
        Bornes de la colorbar.
    cmap : str, default "rainbow"
        Colormap matplotlib.

    Example
    -------
    >>> plot_orbits_by_period(orbites, param="lwp", period_days=5)
    >>> plot_orbits_by_period(orbites, param="iwp", period_days=5, vmax=100)
    """
    from datetime import datetime, timedelta

    if not orbites:
        print("[plot] Aucune orbite a afficher.")
        return

    LABELS = {
        "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
        "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
    }
    title_base, cbar_label = LABELS.get(param, (param.upper(), param))

    # --- Calcul de la date centrale de chaque orbite ------------
    # t0_utc est deja calcule dans prepare_single_orbit() et stocke
    # dans le dict — pas besoin d une epoque arbitraire.

    def _orbit_date(orb):
        t0 = orb.get("t0_utc")
        t  = orb.get("time")
        if t0 is None or t is None:
            return datetime(2025, 12, 1)   # fallback si orbite non preparee
        t_rel = float(np.nanmean(t)) - float(t[0])   # secondes depuis debut orbite
        return t0 + timedelta(seconds=t_rel)

    dates    = [_orbit_date(orb) for orb in orbites]
    t0       = min(dates)
    period_s = period_days * 86400.0

    # Indice de periode pour chaque orbite
    indices  = [int((d - t0).total_seconds() // period_s) for d in dates]
    n_periods = max(indices) + 1

    # Regroupement periode -> liste d'orbites
    groups = {i: [] for i in range(n_periods)}
    for orb, idx in zip(orbites, indices):
        groups[idx].append(orb)

    # --- Une figure par periode ----------------------------------
    for idx in range(n_periods):
        period_orbits = groups[idx]
        if not period_orbits:
            continue

        date_start = t0 + timedelta(days=idx * period_days)
        date_end   = date_start + timedelta(days=period_days - 1)

        period_str = (date_start.strftime("%d/%m/%Y")
                      + " - " + date_end.strftime("%d/%m/%Y"))
        header = ("ECA_JXBA_ACM_CLP_2B\n"
                  + period_str
                  + "  (" + str(len(period_orbits)) + " orbites)")

        fig, ax = _make_polar_map(header)

        gl = ax.gridlines(draw_labels=True, dms=False,
                          x_inline=False, y_inline=False, alpha=0.3,
                          ylocs=np.arange(-90, -55, 10))
        gl.ylabel_style = {"size": 10}
        gl.xlabel_style = {"size": 10}
        gl.top_labels  = False
        gl.right_label = False

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".", markersize=5,
                linestyle="none", label="Dome C", transform=ccrs.PlateCarree())

        sc = None
        for orb in period_orbits:
            values = orb.get(param)
            if values is None:
                continue
            values = np.where(values < 0, np.nan, values)
            sc = ax.scatter(
                orb["lon"], orb["lat"], c=values,
                cmap=cmap, s=5,
                transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax,
            )

        _add_distance_circles(ax)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                                shrink=0.7, pad=0.05)
            cbar.set_label(cbar_label, fontsize=10)

        period_title = (title_base
                        + "  -  periode "
                        + str(idx + 1) + "/" + str(n_periods))
        ax.set_title(period_title, fontsize=10,
                     fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right")
        plt.show()

# ============================================================
# ANALYSE EXPLORATOIRE — orbites brutes
# Les fonctions de cette section acceptent toutes le meme
# format : list[dict] retourne par load_multi_orbits() ou
# merge_orbit_sources().
# Chaque dict contient au minimum : lat, lon, time, lwp, iwp.
# ============================================================

def _collect_param(orbites: list[dict], param: str) -> np.ndarray:
    """Concatene un parametre 1D sur toutes les orbites, sans NaN."""
    parts = []
    for orb in orbites:
        raw = orb.get(param)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).ravel()
        parts.append(arr[~np.isnan(arr) & (arr >= 0)])
    return np.concatenate(parts) if parts else np.array([])


def _collect_two(orbites, p1, p2):
    """Concatene deux parametres 1D en gardant uniquement les paires valides."""
    v1_all, v2_all = [], []
    for orb in orbites:
        a1 = np.asarray(orb.get(p1, []), dtype=float).ravel()
        a2 = np.asarray(orb.get(p2, []), dtype=float).ravel()
        n  = min(len(a1), len(a2))
        if n == 0:
            continue
        a1, a2 = a1[:n], a2[:n]
        valid  = (~np.isnan(a1)) & (~np.isnan(a2)) & (a1 >= 0) & (a2 >= 0)
        v1_all.append(a1[valid])
        v2_all.append(a2[valid])
    if not v1_all:
        return np.array([]), np.array([])
    return np.concatenate(v1_all), np.concatenate(v2_all)


def _orbit_dates(orbites):
    """Retourne la date centrale de chaque orbite sous forme de datetime.

    Utilise t0_utc (deja present dans le dict) + la duree relative
    du tableau time. Pas d epoque arbitraire.
    """
    from datetime import datetime, timedelta
    FALLBACK = datetime(2025, 12, 1)
    dates = []
    for orb in orbites:
        t0 = orb.get("t0_utc")
        t  = orb.get("time")
        if t0 is None or t is None:
            dates.append(FALLBACK)
        else:
            t_rel = float(np.nanmean(t)) - float(t[0])
            dates.append(t0 + timedelta(seconds=t_rel))
    return dates


# ------------------------------------------------------------
# 1. Distribution (histogramme + boxplot)
# ------------------------------------------------------------

def plot_distribution(orbites: list[dict],
                      param: str = "lwp",
                      bins: int = 60,
                      log_scale: bool = True) -> None:
    """Histogramme et boxplot de la distribution de LWP ou IWP.

    Parameters
    ----------
    param : "lwp" ou "iwp"
    bins : nombre de classes de l'histogramme
    log_scale : si True, axe Y en echelle logarithmique
    """
    LABELS = {
        "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
        "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
    }
    long_name, unit_label = LABELS.get(param, (param.upper(), param))

    values = _collect_param(orbites, param)
    if len(values) == 0:
        print(f"[plot] Aucune valeur valide pour {param}.")
        return

    fig, (ax_hist, ax_box) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        long_name + " — distribution  (" + str(len(orbites)) + " orbites)",
        fontsize=13, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    # Histogramme
    ax_hist.hist(values, bins=bins, color=TITLE_COLOR, edgecolor="white",
                 linewidth=0.3, alpha=0.85)
    if log_scale:
        ax_hist.set_yscale("log")
    ax_hist.set_xlabel(unit_label, fontsize=10)
    ax_hist.set_ylabel("Nombre de mesures", fontsize=10)
    ax_hist.set_title("Histogramme", fontsize=10)

    # Statistiques affichees sur le graphe
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    stats_txt = (
        "n     = " + str(len(values)) + "\n"
        + "mean  = " + f"{np.mean(values):.2f}\n"
        + "std   = " + f"{np.std(values):.2f}\n"
        + "med   = " + f"{med:.2f}\n"
        + "Q25   = " + f"{q25:.2f}\n"
        + "Q75   = " + f"{q75:.2f}\n"
        + "max   = " + f"{np.max(values):.2f}"
    )
    ax_hist.text(0.97, 0.97, stats_txt, transform=ax_hist.transAxes,
                 va="top", ha="right", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Boxplot
    bp = ax_box.boxplot(values, patch_artist=True, widths=0.5,
                        medianprops=dict(color="red", linewidth=2))
    bp["boxes"][0].set_facecolor(TITLE_COLOR)
    bp["boxes"][0].set_alpha(0.5)
    ax_box.set_ylabel(unit_label, fontsize=10)
    ax_box.set_title("Boxplot", fontsize=10)
    ax_box.set_xticks([])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 2. Serie temporelle
# ------------------------------------------------------------

def plot_time_series(orbites: list[dict],
                     param: str = "lwp",
                     period_days: int = 5) -> None:
    """Evolution de la moyenne et l'ecart-type par periode de N jours.

    Parameters
    ----------
    param : "lwp" ou "iwp"
    period_days : agregation temporelle en jours
    """
    from datetime import datetime, timedelta

    LABELS = {
        "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
        "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
    }
    long_name, unit_label = LABELS.get(param, (param.upper(), param))

    dates    = _orbit_dates(orbites)
    t0       = min(dates)
    period_s = period_days * 86400.0

    # Regrouper par periode
    groups: dict[int, list] = {}
    for orb, d in zip(orbites, dates):
        idx = int((d - t0).total_seconds() // period_s)
        groups.setdefault(idx, []).append(orb)

    sorted_idx = sorted(groups)
    x_dates, y_mean, y_std, y_med = [], [], [], []

    for idx in sorted_idx:
        vals = _collect_param(groups[idx], param)
        if len(vals) == 0:
            continue
        x_dates.append(t0 + timedelta(days=idx * period_days + period_days / 2))
        y_mean.append(np.mean(vals))
        y_std.append(np.std(vals))
        y_med.append(np.median(vals))

    if not x_dates:
        print(f"[plot] Aucune donnee pour {param}.")
        return

    y_mean = np.array(y_mean)
    y_std  = np.array(y_std)
    y_med  = np.array(y_med)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor("white")

    ax.fill_between(x_dates,
                    y_mean - y_std, y_mean + y_std,
                    color=TITLE_COLOR, alpha=0.2, label="mean +/- std")
    ax.plot(x_dates, y_mean, color=TITLE_COLOR, linewidth=2, marker="o",
            markersize=4, label="Moyenne")
    ax.plot(x_dates, y_med, color="#FFA500", linewidth=1.5,
            linestyle="--", marker="s", markersize=3, label="Mediane")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(unit_label, fontsize=10)
    ax.set_title(
        long_name + " — serie temporelle (pas " + str(period_days) + " jours)",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3. Correlation LWP vs IWP
# ------------------------------------------------------------

def plot_correlation_lwp_iwp(orbites: list[dict],
                              max_points: int = 50_000) -> None:
    """Nuage de points LWP vs IWP avec densite de couleur.

    Parameters
    ----------
    max_points : sous-echantillonnage aleatoire si trop de points
                 (evite un affichage trop lent)
    """
    lwp, iwp = _collect_two(orbites, "lwp", "iwp")
    if len(lwp) == 0:
        print("[plot] Aucune paire LWP/IWP valide.")
        return

    # Sous-echantillonnage si necessaire
    if len(lwp) > max_points:
        idx = np.random.choice(len(lwp), max_points, replace=False)
        lwp, iwp = lwp[idx], iwp[idx]

    # Densite 2D pour colorier les points
    from scipy.stats import gaussian_kde
    try:
        xy  = np.vstack([lwp, iwp])
        kde = gaussian_kde(xy)
        z   = kde(xy)
        order = np.argsort(z)
        lwp, iwp, z = lwp[order], iwp[order], z[order]
        c = z
    except Exception:
        c = TITLE_COLOR

    # Correlation de Pearson
    r = float(np.corrcoef(lwp, iwp)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")

    sc = ax.scatter(lwp, iwp, c=c, cmap="plasma", s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Densite")

    # Droite de regression
    coeffs   = np.polyfit(lwp, iwp, 1)
    x_line   = np.linspace(lwp.min(), lwp.max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color="red", linewidth=1.5, linestyle="--",
            label="Regression  r=" + f"{r:.3f}")

    ax.set_xlabel("LWP ($g/m^2$)", fontsize=10)
    ax.set_ylabel("IWP ($g/m^2$)", fontsize=10)
    ax.set_title("Correlation LWP vs IWP",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. Variation selon la latitude
# ------------------------------------------------------------

def plot_latitudinal_profile(orbites: list[dict],
                              param: str = "lwp",
                              dlat: float = 2.0) -> None:
    """Profil latitudinal : moyenne et ecart-type par bande de latitude.

    Parameters
    ----------
    param : "lwp" ou "iwp"
    dlat : largeur des bandes de latitude en degres
    """
    LABELS = {
        "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
        "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
    }
    long_name, unit_label = LABELS.get(param, (param.upper(), param))

    # Concatener lat + param sur toutes les orbites
    lat_all, val_all = _collect_two(orbites, "lat", param)
    if len(lat_all) == 0:
        print(f"[plot] Aucune donnee valide pour {param}.")
        return

    # Bandes de latitude
    lat_min = np.floor(lat_all.min() / dlat) * dlat
    lat_max = np.ceil(lat_all.max()  / dlat) * dlat
    edges   = np.arange(lat_min, lat_max + dlat, dlat)
    centers = (edges[:-1] + edges[1:]) / 2

    means, stds, medians, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lat_all >= lo) & (lat_all < hi)
        v    = val_all[mask]
        if len(v) < 2:
            means.append(np.nan)
            stds.append(np.nan)
            medians.append(np.nan)
            counts.append(0)
        else:
            means.append(np.mean(v))
            stds.append(np.std(v))
            medians.append(np.median(v))
            counts.append(len(v))

    means   = np.array(means)
    stds    = np.array(stds)
    medians = np.array(medians)
    counts  = np.array(counts)

    fig, (ax_main, ax_count) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        long_name + " — profil latitudinal (pas " + str(dlat) + "deg)",
        fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
    )

    # Profil principal
    ax_main.fill_betweenx(centers,
                           means - stds, means + stds,
                           color=TITLE_COLOR, alpha=0.2, label="mean +/- std")
    ax_main.plot(means,   centers, color=TITLE_COLOR, linewidth=2,
                 marker="o", markersize=4, label="Moyenne")
    ax_main.plot(medians, centers, color="#FFA500", linewidth=1.5,
                 linestyle="--", marker="s", markersize=3, label="Mediane")
    ax_main.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    ax_main.set_xlabel(unit_label, fontsize=10)
    ax_main.set_ylabel("Latitude (deg)", fontsize=10)
    ax_main.legend(fontsize=9)
    ax_main.grid(alpha=0.3)

    # Nombre de mesures par bande
    ax_count.barh(centers, counts, height=dlat * 0.8,
                  color="#457B9D", alpha=0.7)
    ax_count.set_xlabel("Nb mesures", fontsize=9)
    ax_count.set_title("Echantillon", fontsize=9)
    ax_count.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5. Tableau de bord complet
# ------------------------------------------------------------

def plot_eda(orbites: list[dict], period_days: int = 5) -> None:
    """Lance toutes les analyses exploratoires en une seule commande.

    Parameters
    ----------
    period_days : pas de temps pour la serie temporelle
    """
    print("=== Distribution LWP ===")
    plot_distribution(orbites, param="lwp")
    print("=== Distribution IWP ===")
    plot_distribution(orbites, param="iwp")
    print("=== Serie temporelle LWP ===")
    plot_time_series(orbites, param="lwp", period_days=period_days)
    print("=== Serie temporelle IWP ===")
    plot_time_series(orbites, param="iwp", period_days=period_days)
    print("=== Correlation LWP vs IWP ===")
    plot_correlation_lwp_iwp(orbites)
    print("=== Profil latitudinal LWP ===")
    plot_latitudinal_profile(orbites, param="lwp")
    print("=== Profil latitudinal IWP ===")
    plot_latitudinal_profile(orbites, param="iwp")

# ------------------------------------------------------------
# Statistiques descriptives
# ------------------------------------------------------------

def descriptive_stats(orbites: list[dict],
                      params: list[str] | None = None) -> "pd.DataFrame":
    """Calcule et affiche les statistiques descriptives de LWP et IWP
    (ou tout autre parametre 1D) sur l ensemble des orbites.

    Parameters
    ----------
    orbites : list[dict]
        Orbites brutes (sortie de load_multi_orbits / merge_orbit_sources).
    params : list[str], optional
        Parametres a analyser. Par defaut : ["lwp", "iwp"].

    Returns
    -------
    pandas.DataFrame  avec une ligne par parametre.
    Les colonnes sont : n, min, max, mean, std, median, Q25, Q75, % zeros.

    Example
    -------
    >>> df = descriptive_stats(all_raw)
    >>> df = descriptive_stats(all_raw, params=["lwp", "iwp", "surface_elevation"])
    """
    import pandas as pd

    if params is None:
        params = ["lwp", "iwp"]

    rows = []
    for param in params:
        values = _collect_param(orbites, param)

        if len(values) == 0:
            rows.append({
                "parametre": param,
                "n":         0,
                "min":       np.nan,
                "max":       np.nan,
                "mean":      np.nan,
                "std":       np.nan,
                "median":    np.nan,
                "Q25":       np.nan,
                "Q75":       np.nan,
                "% zeros":   np.nan,
            })
            continue

        q25, med, q75 = np.percentile(values, [25, 50, 75])

        rows.append({
            "parametre": param,
            "n":         len(values),
            "min":       float(np.min(values)),
            "max":       float(np.max(values)),
            "mean":      float(np.mean(values)),
            "std":       float(np.std(values)),
            "median":    float(med),
            "Q25":       float(q25),
            "Q75":       float(q75),
            "% zeros":   float(100.0 * np.sum(values == 0) / len(values)),
        })

    df = pd.DataFrame(rows).set_index("parametre")

    # Arrondi pour la lisibilite
    df = df.round(3)

    # Affichage dans le terminal ou Jupyter
    print("\n" + "=" * 60)
    print("  Statistiques descriptives")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60 + "\n")

    return df

def describe_lwp_iwp(orbites: list[dict],
                     params: list[str] | None = None) -> "pd.DataFrame":
    """Construit un DataFrame pandas avec LWP et IWP puis appelle describe().

    Parameters
    ----------
    orbites : list[dict]
        Orbites brutes (sortie de load_multi_orbits / merge_orbit_sources).
    params : list[str], optional
        Parametres a analyser. Par defaut : ["lwp", "iwp"].

    Returns
    -------
    pandas.DataFrame  retourne par describe() — affiche automatiquement
    dans Jupyter sous forme de tableau HTML.

    Example
    -------
    >>> df = describe_lwp_iwp(all_raw)
    >>> df = describe_lwp_iwp(all_raw, params=["lwp", "iwp", "surface_elevation"])
    """
    import pandas as pd

    if params is None:
        params = ["lwp", "iwp"]

    # Construire le DataFrame : une colonne par parametre
    # On tronque toutes les colonnes a la longueur minimale pour aligner les lignes
    series = {}
    for param in params:
        values = _collect_param(orbites, param)
        series[param] = values

    # Alignement : longueur minimale commune
    n_min = min(len(v) for v in series.values()) if series else 0
    df = pd.DataFrame({p: v[:n_min] for p, v in series.items()})

    print("\nDataFrame shape :", df.shape)
    stats = df.describe()
    print(stats.to_string())
    return stats

# ------------------------------------------------------------
# Statistiques descriptives par orbite + filtre sur seuil
# ------------------------------------------------------------

def describe_by_orbit(orbites: list[dict],
                      param: str = "lwp") -> "pd.DataFrame":
    """Statistiques descriptives par orbite via pandas describe().

    Construit un DataFrame avec une colonne par orbite identifiee
    par son orbit_id, puis appelle describe() pour obtenir :
    count, mean, std, min, 25%, 50%, 75%, max.

    Parameters
    ----------
    param : str  parametre a analyser : "lwp" ou "iwp"

    Returns
    -------
    pandas.DataFrame  (8 lignes x n_orbites colonnes)
    """
    import pandas as pd

    series = {}
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        col_id = orb.get("orbit_id") or orb.get("nom_orbite", "unknown")
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr) & (arr >= 0)]
        series[col_id] = pd.Series(arr)

    if not series:
        print("[stats] Aucune donnee pour " + param)
        return pd.DataFrame()

    df    = pd.DataFrame(series)
    stats = df.describe()

    print("\n" + "=" * 60)
    print("  Statistiques " + param.upper() + " par orbite")
    print("=" * 60)
    print(stats.to_string())
    print("=" * 60 + "\n")

    return stats


def orbites_above_threshold(orbites: list[dict],
                             param: str = "lwp",
                             threshold: float = 100.0,
                             plot: bool = True) -> list[dict]:
    """Identifie et trace les orbites ayant au moins une valeur > seuil.

    Parameters
    ----------
    param : str          parametre : "lwp" ou "iwp"
    threshold : float    seuil en g/m2 (defaut 100)
    plot : bool          trace une carte polaire si True

    Returns
    -------
    list[dict]  sous-liste des orbites depassant le seuil

    Example
    -------
    >>> filtered = orbites_above_threshold(orbites, param="lwp", threshold=100)
    >>> filtered = orbites_above_threshold(orbites, param="iwp", threshold=100)
    """
    import pandas as pd

    # --- DataFrame : une colonne par orbite --------------------
    series = {}
    for orb in orbites:
        values = orb.get(param)
        if values is None:
            continue
        col_id = orb.get("orbit_id") or orb.get("nom_orbite", "unknown")
        arr = np.asarray(values, dtype=float)
        series[col_id] = pd.Series(arr)

    if not series:
        print("[filtre] Aucune donnee pour " + param)
        return []

    df = pd.DataFrame(series)

    # Colonnes avec au moins une valeur > seuil
    cols_above = df.columns[(df > threshold).any()].tolist()

    print("\n[filtre] " + param.upper() + " > " + str(threshold) + " g/m2")
    print("  " + str(len(cols_above)) + "/" + str(len(orbites)) + " orbite(s) concernee(s) :")
    for col in cols_above:
        n_above  = int((df[col] > threshold).sum())
        max_val  = float(df[col].max())
        orb_info = next((o for o in orbites
                         if (o.get("orbit_id") or o.get("nom_orbite")) == col), {})
        date_str = orb_info.get("date_debut", "")
        print("  -> " + col
              + "  |  date: " + date_str
              + "  |  max=" + f"{max_val:.1f}"
              + "  |  n>" + str(threshold) + ": " + str(n_above))

    if not cols_above:
        print("  Aucune orbite ne depasse ce seuil.")
        return []

    filtered = [
        orb for orb in orbites
        if (orb.get("orbit_id") or orb.get("nom_orbite")) in cols_above
    ]

    # --- Carte polaire : contexte gris + orbites filtrees en couleur ---
    if plot:
        LABELS = {
            "lwp": ("Liquid Water Path", "LWP ($g/m^2$)"),
            "iwp": ("Ice Water Path",    "IWP ($g/m^2$)"),
        }
        long_name, cbar_label = LABELS.get(param, (param.upper(), param))
        vmax_plot = float(df[cols_above].max().max())

        header = ("ECA_JXBA_ACM_CLP_2B\n"
                  + param.upper() + " > " + str(threshold) + " g/m2  "
                  + "(" + str(len(filtered)) + "/" + str(len(orbites)) + " orbites)")

        fig, ax = _make_polar_map(header)
        ax.gridlines(draw_labels=True, dms=False,
                     x_inline=False, y_inline=False,
                     alpha=0.3, ylocs=np.arange(-90, -55, 10))
        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                label="Dome C", transform=ccrs.PlateCarree())

        # Toutes les orbites en gris (contexte)
        for orb in orbites:
            if orb.get("param") is None:
                pass
            ax.scatter(orb["lon"], orb["lat"],
                       color="lightgrey", s=2, alpha=0.4,
                       transform=ccrs.PlateCarree(), zorder=1)

        # Orbites filtrees en couleur
        sc = None
        for orb in filtered:
            values = np.asarray(orb.get(param, []), dtype=float)
            values = np.where(values < 0, np.nan, values)
            sc = ax.scatter(orb["lon"], orb["lat"],
                            c=values, cmap="rainbow", s=6,
                            vmin=0, vmax=vmax_plot,
                            transform=ccrs.PlateCarree(), zorder=5)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                                shrink=0.7, pad=0.05)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.axhline(y=threshold, color="red",
                            linewidth=1.5, linestyle="--")

        _add_distance_circles(ax)
        ax.set_title(
            long_name + "  —  orbites avec "
            + param.upper() + " > " + str(threshold) + " g/m2",
            fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR,
        )
        ax.legend(loc="upper right")
        plt.show()

    return filtered