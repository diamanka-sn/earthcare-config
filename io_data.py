# ============================================================
# io_data.py
# Téléchargement EarthCARE et lecture des fichiers HDF5.
# ============================================================

import h5py
import earthcarekit as eck

from config import FILE_TYPE, HDF5_FIELDS


def download_product(orbit_and_frame: str, start_time: str, end_time: str) -> None:
    """Télécharge un produit EarthCARE pour une orbite et une période données."""
    eck.ecdownload(
        file_type=FILE_TYPE,
        orbit_and_frame=orbit_and_frame,
        start_time=start_time,
        end_time=end_time,
    )


def download_multi_period(periods: list[tuple[str, str]], frame_id: str = "G") -> None:
    """Télécharge le produit sur plusieurs périodes (mode multi-orbites).

    Parameters
    ----------
    periods : list of (start_time, end_time)
    frame_id : str, default "G"
    """
    for start, end in periods:
        eck.ecdownload(
            file_type=FILE_TYPE,
            frame_id=frame_id,
            start_time=start,
            end_time=end,
        )


def search_product(start_time: str, end_time: str,
                   orbit_and_frame: str | None = None,
                   frame_id: str | None = None):
    """Recherche les fichiers produit disponibles.

    Returns
    -------
    Dataset earthcarekit avec attribut `filepath`.
    """
    kwargs = dict(file_type=FILE_TYPE, start_time=start_time, end_time=end_time)
    if orbit_and_frame:
        kwargs["orbit_and_frame"] = orbit_and_frame
    if frame_id:
        kwargs["frame_id"] = frame_id
    return eck.search_product(**kwargs)


def load_orbit(filepath: str, extra_fields: dict | None = None) -> dict:
    """Lit un fichier HDF5 EarthCARE et retourne un dict de tableaux numpy.

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier .h5
    extra_fields : dict, optional
        Champs supplémentaires {nom: chemin_hdf5} à charger en complément
        des champs standard définis dans ``config.HDF5_FIELDS``.

    Returns
    -------
    dict
        {nom_variable: ndarray}
    """
    fields = {**HDF5_FIELDS, **(extra_fields or {})}
    data = {}
    with h5py.File(filepath, "r") as f:
        for name, path in fields.items():
            data[name] = f[path][:]
    return data


def load_multi_orbits(filepaths, extra_fields: dict | None = None) -> list[dict]:
    """Charge une liste de fichiers orbite en gérant les erreurs.

    Parameters
    ----------
    filepaths : iterable of str
    extra_fields : dict, optional
        Voir ``load_orbit``.

    Returns
    -------
    list of dict  (orbites chargées avec succès)
    """
    orbites = []
    for fp in filepaths:
        try:
            data = load_orbit(fp, extra_fields=extra_fields)
            orbites.append(data)
            print(f"Orbite {data.get('nom_orbite', fp)} chargée")
        except Exception as e:
            print(f"Erreur {fp} : {e}")
    return orbites


def get_t0_utc(filepath: str, fallback=None):
    """Extrait l'heure UTC de début de passage depuis l'en-tête HDF5.

    Parameters
    ----------
    fallback : datetime, optional
        Valeur de repli si la lecture échoue.
    """
    from datetime import datetime
    try:
        with h5py.File(filepath, "r") as f:
            raw = f["HeaderData/VariableProductHeader/MainProductHeader/sensingStartTime"][()]
            s = raw.decode().replace("UTC=", "").replace("Z", "")
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return fallback or datetime(2025, 12, 30, 21, 50, 0)
