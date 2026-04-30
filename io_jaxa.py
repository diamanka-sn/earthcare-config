# ============================================================
# io_jaxa.py
# Chargement des fichiers EarthCARE téléchargés depuis le portail
# JAXA. La structure HDF5 interne étant identique à celle des
# fichiers ESA, on réutilise directement load_orbit().
#
# Ce module se limite donc à :
#   - load_jaxa_orbits(dir)  → charge tous les .h5 d'un dossier
#   - merge_orbit_sources()  → fusionne et trie ESA + JAXA
# ============================================================

from pathlib import Path

import numpy as np

from io_data import load_multi_orbits


def load_jaxa_orbits(directory: str,
                     extra_fields: dict | None = None) -> list[dict]:
    """Charge tous les fichiers .h5 présents dans un dossier JAXA.

    Délègue entièrement à ``load_orbit()`` — même structure HDF5,
    mêmes chemins, même dict retourné.

    Parameters
    ----------
    directory : str
        Dossier contenant les fichiers .h5 JAXA.
    extra_fields : dict, optional
        Champs HDF5 supplémentaires à lire (transmis à load_orbit).

    Returns
    -------
    list[dict]  — même format que load_multi_orbits()
    """
    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        print(f"[JAXA] Aucun fichier .h5 trouvé dans : {directory}")
        return []

    print(f"[JAXA] {len(files)} fichier(s) trouvé(s) dans {directory}")
    return load_multi_orbits(files, extra_fields=extra_fields)


def merge_orbit_sources(esa_orbits: list[dict],
                        jaxa_orbits: list[dict],
                        sort_by: str = "time") -> list[dict]:
    """Fusionne et trie chronologiquement des orbites ESA et JAXA.

    Les deux listes contiennent des dicts de même structure (produit
    de load_orbit / load_jaxa_orbits). Le tri s'effectue sur la valeur
    minimale de la clé ``sort_by`` (temps absolu en secondes).

    Parameters
    ----------
    sort_by : str, default "time"
        Clé du dict utilisée pour l'ordre chronologique.

    Returns
    -------
    list[dict]  triée chronologiquement
    """
    def _sort_key(orb):
        arr = orb.get(sort_by)
        return float(np.nanmin(arr)) if arr is not None else 0.0

    merged = sorted(esa_orbits + jaxa_orbits, key=_sort_key)
    print(f"[merge] {len(esa_orbits)} orbite(s) ESA  +  {len(jaxa_orbits)} orbite(s) JAXA"
          f"  →  {len(merged)} au total, triées par '{sort_by}'.")
    return merged


from datetime import datetime

def merge_orbit_sources(esa_orbits: list[dict],
                        jaxa_orbits: list[dict],
                        sort_by: str = "start_time") -> list[dict]:

    def _parse_start_time(val) -> datetime:
        """Décode np.bytes_(b'UTC=2025-12-02T01:17:31') → datetime."""
        if val is None:
            return datetime.min
        # Décoder bytes si nécessaire
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode("utf-8")
        # Supprimer le préfixe "UTC="
        val = val.removeprefix("UTC=")
        return datetime.fromisoformat(val)

    def _sort_key(orb):
        return _parse_start_time(orb.get(sort_by))

    merged = sorted(esa_orbits + jaxa_orbits, key=_sort_key)
    print(f"[merge] {len(esa_orbits)} orbite(s) ESA  +  {len(jaxa_orbits)} orbite(s) JAXA"
          f"  →  {len(merged)} au total, triées par '{sort_by}'.")
    return merged