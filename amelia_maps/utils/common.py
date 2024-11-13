import os
import contextily as ctx

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, "../..")
ROOT_DIR = os.path.normpath(ROOT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

AIRPORTS_PATH = os.path.join(ROOT_DIR, "amelia_maps/utils/airports.txt")

DATA_DIR = os.path.join(ROOT_DIR, "datasets/amelia")
VERSION = "a10v08"


MAP_PROVIDERS = {
    'positron': ctx.providers.CartoDB.Positron,
    'darkmatter': ctx.providers.CartoDB.DarkMatter,
    'voyager': ctx.providers.CartoDB.Voyager,
    'mapnik': ctx.providers.OpenStreetMap.Mapnik,
    'humanitarian': ctx.providers.OpenStreetMap.HOT,
    'cyclosm': ctx.providers.CyclOSM,
    'stadia_outdoors': ctx.providers.Stadia.Outdoors,
}
