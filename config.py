# ============================================================
# config.py
# Toutes les constantes du projet ECA_JXBA_ACM_CLP_2B.
# Modifier ce fichier pour changer d'orbite, de station ou
# de période d'analyse — aucun autre fichier à toucher.
# ============================================================

# --- Produit EarthCARE -------------------------------------------
FILE_TYPE   = "ACM_CLP_2B"
ORBIT_FRAME = "09039G"
DATE_START  = "2025-12-30"
DATE_END    = "2025-12-31"

# Périodes de téléchargement multi-orbites (liste de tuples)
DOWNLOAD_PERIODS = [
    ("2025-11-01", "2025-12-01"),
    ("2025-12-01", "2026-02-01"),
]

# --- Fenêtre temporelle d'analyse (s depuis début orbite) --------
T_MIN = 440
T_MAX = 580

# --- Station de référence — Concordia / Dome C -------------------
LAT_REF = -75.1
LON_REF = 123.35

# --- Style graphique ---------------------------------------------
TITLE_COLOR  = "#003399"
TITLE_WEIGHT = "bold"

# Cercles de distance autour de la station
CIRCLES_CONFIG = [
    {"radius": 500,  "color": "#E63946", "label": "500 km",  "linestyle": "-"},
    {"radius": 1000, "color": "#457B9D", "label": "1000 km", "linestyle": "-"},
]

# --- Nomenclature des types de particules ------------------------
PARTICLE_LABEL = {
    0:  "clear",
    1:  "warm water",
    2:  "Supercooled water",
    3:  "3d ice",
    4:  "2d plate",
    5:  "Mix 3D ice and 2D plate",
    6:  "liquid drizzle",
    7:  "Mixed phase drizzle",
    8:  "Rain",
    9:  "snow",
    10: "water + liquid drizzle",
    11: "water + rain",
    12: "Mixed Phase",
    13: "unknown",
}

PARTICLE_COLORS = {
    0:  "#ffffff",
    1:  "#2196F3",
    2:  "#0009b0",
    3:  "#02ab24",
    4:  "#9C27B0",
    5:  "#7B1FA2",
    6:  "#80DEEA",
    7:  "#00ACC1",
    8:  "#FF9890",
    9:  "#B0BEC5",
    10: "#0288D1",
    11: "#E65100",
    12: "#b53c00",
    13: "#310202",
}

# --- Chemins HDF5 (ScienceData) ----------------------------------
HDF5_FIELDS = {
    "particle_type":     "ScienceData/Data/cloud_particle_type_cpr_atlid_msi_1km",
    "temperature":       "ScienceData/Data/GRID_temperature_1km",
    "iwc":               "ScienceData/Data/cloud_ice_content_1km",
    "iwp":               "ScienceData/Data/cloud_ice_water_path_1km",
    "lwc":               "ScienceData/Data/cloud_water_content_1km",
    "lwp":               "ScienceData/Data/cloud_water_path_1km",
    "lat":               "ScienceData/Geo/latitude",
    "lon":               "ScienceData/Geo/longitude",
    "height":            "ScienceData/Geo/height",
    "time":              "ScienceData/Geo/time",
    "surface_elevation": "ScienceData/Geo/surface_elevation",
}

# Champs supplémentaires pour le mode multi-orbites
HDF5_FIELDS_ORBIT_META = {
    "nom_orbite": "HeaderData/VariableProductHeader/MainProductHeader/orbitNumber",
    "frame_id":   "HeaderData/VariableProductHeader/MainProductHeader/frameID",
}
