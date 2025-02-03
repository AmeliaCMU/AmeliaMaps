import json
import math
import os
import pandas as pd
import yaml
import contextily as ctx
from easydict import EasyDict
from tqdm import tqdm
from geopy.geocoders import Nominatim


from amelia_maps.utils import common as C
from amelia_maps.utils import utils as U


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(self.variance())


def fetch_location_coordinates(location):
    geolocator = Nominatim(user_agent="my-app")
    response = geolocator.geocode(location)
    bbox = response.raw['boundingbox']
    return response, bbox


def run_limits(base_dir: str, airport: str, traj_version: str, output: str):

    black_list = ['Frame', 'ID', 'Type', 'Interp']

    traj_data_dir = os.path.join(base_dir, f'traj_data_{traj_version}', 'raw_trajectories', f'{airport}')
    assets_dir = os.path.join(base_dir, 'assets', f'{airport}')

    limits_file = os.path.join(assets_dir, 'limits.json')
    if not os.path.exists(limits_file):
        print(f"\tLimits file not found for {airport}. Fetching location coordinates...")
        try:
            response, (south, north, west, east) = fetch_location_coordinates(airport)
        except:
            print(f"Bad request, for airport {airport}. Skipping...")
            return
        ll_limits = (float(north), float(east), float(south), float(west))
        airport_name = components = [comp.strip() for comp in response[0].split(',')][0]
        latlng = response[1]
        U.create_limits(ll_limits, (airport, airport_name), latlng, output)

        limits_file = os.path.join(output, 'limits.json')

    print(f"\tFound limits file: {limits_file}")
    with open(limits_file, 'r') as f:
        ref_data = EasyDict(json.load(f))

    # print(json.dumps(ref_data.limits.Altitude, indent=4))

    traj_files = [os.path.join(traj_data_dir, f) for f in os.listdir(traj_data_dir)]
    print(f"\tFound {len(traj_files)} trajectory files in {traj_data_dir}")

    data = pd.read_csv(traj_files[0])

    limits = {}
    incstats = {}
    for k, v in data.items():
        if k in black_list:
            continue
        limits[k] = {
            "min": float('inf'), "max": -float('inf'), "mean": 0.0, "std": 0.0
        }
        incstats[k] = RunningStats()

    for f in tqdm(traj_files):
        data = pd.read_csv(traj_files[0])
        for k in limits.keys():
            arr = data[k].to_numpy()
            limits[k]["min"] = min(limits[k]["min"], arr.min())
            limits[k]["max"] = max(limits[k]["max"], arr.max())

            for a in arr:
                incstats[k].push(a)

        for k in limits.keys():
            limits[k]["mean"] = incstats[k].mean()
            limits[k]["std"] = incstats[k].std()

    ref_data['limits'] = limits
    with open(limits_file, 'w') as f:
        json.dump(ref_data, f)
    print(f"\tAdding limits to reference file in: {limits_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input directory')
    parser.add_argument('--airport', default='ksea', type=str, help='ICAO Airport Code')
    parser.add_argument('--traj_version', default=C.VERSION, type=str, help='Input directory')
    parser.add_argument('--output', default=C.OUTPUT_DIR, type=str, help='Output directory')
    args = parser.parse_args()
    airports = U.get_airport_list(args.traj_version)
    if args.airport != "all" and args.airport not in airports:
        raise ValueError(f"Invalid airport code: {args.airport} there aren't raw trajectories for this airport")

    if args.airport == "all":
        airports = U.get_airport_list(args.traj_version)
    else:
        airports = [args.airport]

    kargs = vars(args)

    for airport in tqdm(airports):
        print(f"Running: {airport}")
        kargs['airport'] = airport
        output = os.path.join(kargs['output'], airport)
        os.makedirs(output, exist_ok=True)
        kargs['output'] = output
        run_limits(**kargs)
