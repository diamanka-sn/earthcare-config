# ============================================================
# processing.py
# Calculs scientifiques : géodésie, masques, temps solaire,
# et préparation des tableaux pour les figures.
# ============================================================

from datetime import timedelta

import numpy as np

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

def prepare_multi_orbits(raw_orbits: list[dict]) -> list[dict]:
    """Nettoie la liste brute des orbites pour l'affichage multi-traces.

    Retourne une liste de dicts avec les clés : nom_orbite, lat, lon, lwp, iwp.
    """
    result = []
    for data in raw_orbits:
        result.append({
            "nom_orbite": data["nom_orbite"],
            "lat":        data["lat"],
            "lon":        data["lon"],
            "lwp":        mask_negative(data["lwp"]),
            "iwp":        mask_negative(data["iwp"]),
        })
    return result


def build_orbit_label(orbites: list[dict]) -> str:
    """Construit la chaîne lisible des numéros d'orbite pour les titres."""
    if not orbites:
        return ""
    last = set(orbites[-1])
    frame_id  = last.get("frame_id", b"G")
    frame_str = str(frame_id).split("'")[1] if "'" in str(frame_id) else str(frame_id)
    return ", ".join(str(orb["nom_orbite"][0]) + frame_str for orb in orbites)

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