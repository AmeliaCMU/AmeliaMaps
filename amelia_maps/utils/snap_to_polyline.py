import os
from typing import Tuple
import osmnx as ox
from math import floor
import pandas as pd
from tqdm import tqdm
import multiprocessing
from amelia_maps.graph_processor import MapProcessor


class TrajectoryProcessor(MapProcessor):
    def __init__(self, ipath: str, opath: str, airport: str, parallel: bool, data_to_process: int = 0.05):
        super().__init__(ipath, opath, airport)
        self.data_percentage = data_to_process
        self.parrallel = parallel
        self.TRAJECTORIES_DIR = os.path.join(self.BASE_DIR, 'raw_trajectories', self.AIRPORT)
        self.TRAJECTORY_FILES = [f for f in os.listdir(self.TRAJECTORIES_DIR)]

    def get_polyline(self, ll):
        lon, lat = ll
        nearest_node = ox.nearest_nodes(self.graph, lon, lat, return_dist=False)
        graph_lon = self.graph._node[nearest_node]['x']
        graph_lat = self.graph._node[nearest_node]['y']
        x, y = self.calculate_x_y((graph_lon, graph_lat), True)
        return graph_lon, graph_lat, x, y

    def process_file_sequential(self, file, out_path_set, save=True):
        in_file = os.path.join(self.TRAJECTORIES_DIR, file)
        data = pd.read_csv(in_file)
        _Lon, _Lat = data.Lon.to_numpy(), data.Lat.to_numpy()
        Node_Lon = []
        Node_Lat = []
        X = []
        Y = []
        assert len(_Lon) == len(_Lat)
        for ll in zip(_Lon, _Lat):
            graph_lon, graph_lat, x, y = self.get_polyline(ll)
            Node_Lon.append(graph_lon)
            Node_Lat.append(graph_lat)
            X.append(x)
            Y.append(y)
        data['node_lon'] = Node_Lon
        data['node_lat'] = Node_Lat
        data['node_x'] = X
        data['node_y'] = Y
        out_file = os.path.join(out_path_set, file)
        if save:
            data.to_csv(out_file, index=False)

    def process_file_parrallel(self, file, out_path_set, save=True):
        in_file = os.path.join(self.TRAJECTORIES_DIR, file)
        data = pd.read_csv(in_file)
        _Lon, _Lat = data.Lon.to_numpy(), data.Lat.to_numpy()
        assert len(_Lon) == len(_Lat)
        pool = multiprocessing.Pool(processes=10)
        result = pool.map(self.get_polyline, zip(_Lon, _Lat))
        pool.close()
        pool.join()
        Node_Lon, Node_Lat, X, Y = zip(*result)
        data['node_x'] = X
        data['node_y'] = Y
        data['node_lon'] = Node_Lon
        data['node_lat'] = Node_Lat
        out_file = os.path.join(out_path_set, file)
        if save:
            data.to_csv(out_file, index=False)

    def snap_to_graph(self):
        out_path_set = os.path.join(self.BASE_DIR, 'raw_trajectories_snap', self.AIRPORT)
        self.preprocess_map()
        data_length = floor(len(self.TRAJECTORY_FILES) * self.data_percentage)
        for i in tqdm(range(0, data_length)):
            if (self.parrallel):
                self.process_file_parrallel(self.TRAJECTORY_FILES[i], out_path_set)
            else:
                self.process_file_sequential(self.TRAJECTORY_FILES[i], out_path_set)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='./boeing/swim/', type=str, help='Input path.')
    parser.add_argument('--opath', default='./boeing/airports/context',
                        type=str, help='Output path.')
    parser.add_argument('--airport', default='kewr', type=str, help='Airport to process.')
    parser.add_argument('--parallel', default=True, type=str, help='Enable parrallel computing')
    parser.add_argument('--data-to-process', default=0.05, type=float,
                        help='Decimal percentage of dataset to process')
    args = parser.parse_args()

    processor = TrajectoryProcessor(**vars(args))
    processor.snap_to_graph()
