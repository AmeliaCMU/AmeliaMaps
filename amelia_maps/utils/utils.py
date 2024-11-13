import os
import glob
import json
from pyproj import Transformer

from amelia_maps.utils import common as C


def get_airport_list(traj_version: str = '') -> list:
    if not traj_version:
        assets_dir = os.path.join(C.DATA_DIR, 'assets')
    else:
        assets_dir = os.path.join(C.DATA_DIR, f'traj_data_{traj_version}/raw_trajectories')
    files = glob.glob(f"{assets_dir}/*")
    airport_list = []
    for file in files:
        if os.path.isdir(file) and file != "blacklist":
            airport_list.append(os.path.basename(file))
    return airport_list


def transform_extent(extent, original_crs: str, target_crs: str):
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)
    north, east, south, west = extent
    xmin_trans, ymin_trans = transformer.transform(west, south)
    xmax_trans, ymax_trans = transformer.transform(east, north)
    return (ymax_trans, xmax_trans, ymin_trans, xmin_trans)


def create_limits(limits, airport_metadata, latlng, output):
    ref_lat, ref_lon = latlng
    airport_code, airport_name = airport_metadata
    ll_limits = transform_extent(limits, original_crs='EPSG:3857', target_crs='EPSG:4326')

    json_data = {
        "airport_name": airport_name,
        "airport_id": airport_code,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "range_scale": 1000.0,
        "ll_offset": 0.002,

        "espg_4326": {
            "north": ll_limits[0],
            "east": ll_limits[1],
            "south": ll_limits[2],
            "west": ll_limits[3]
        },

        "espg_3857": {
            "north": limits[0],
            "east": limits[1],
            "south": limits[2],
            "west": limits[3]
        }
    }

    with open(f'{output}/limits.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
        print(f"Created limits file for {airport_code}")
