import os
import json
import osmnx as ox
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle as pkl
from geographiclib.geodesic import Geodesic


from amelia_maps.utils.graph_utils import print_stats, GEOD, get_new_endpoint, COLOR_CODES, correct_id, edge_length_total_geod, calculate_x_y, ROOT_DIR


class MapProcessor():
    def __init__(self, base_dir: str, output_dir: str, airport: str, save: bool = True, show: bool = False):
        # Specify paths for necessary files and output directory.
        self.airport = airport
        self.base_dir = base_dir
        self.output_dir = os.path.join(output_dir, self.airport)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.assets_dir = os.path.join(self.base_dir, 'assets', self.airport)

        self.map_dir = os.path.join(self.assets_dir, f'{self.airport}.osm')
        # in case the map is not found, use the map from the network
        self.map_dir = os.path.join(self.assets_dir, f'{self.airport}_from_net.osm')

        limits_file = os.path.join(self.assets_dir, 'limits.json')

        self.geodesic = Geodesic.WGS84  # geoide of the Earth
        with open(limits_file, 'r') as f:
            self.ref_data = json.load(f)
        self.reference_point = (self.ref_data['ref_lon'], self.ref_data['ref_lat'])
        # process the abstract node classes to classify nodes.
        node_path = os.path.join(self.base_dir, '../..', "amelia_maps/utils/node_types.txt")
        node_path = os.path.normpath(node_path)
        node_file = open(node_path, "r")
        self.node_type = node_file.read().split(',')
        # Create dictionaries used for processing.
        map_info = {}
        map_info['all_polylines'] = np.array([])
        self.class_hash = {}  # Hash to convert from string value to numerical value.
        self.inv_class_hash = {}  # Hash to convert from numerical value to string.
        self.scenario = {}
        self.scenario['hold_lines'] = []
        # Populate dictionaries based on the reported node types
        for i in range(0, len(self.node_type)):
            val = i
            if (i != 0):
                map_info[self.node_type[i]] = []
                self.inv_class_hash[val] = self.node_type[i]
            self.class_hash[self.node_type[i]] = val
        self.scenario['map_infos'] = map_info
        self.tags = ['x', 'y']
        self.runway_pairs = []
        # Load graph from XML file, and do preliminary clean-up.
        self.graph = ox.graph_from_xml(self.map_dir, retain_all=True,
                                       simplify=False)  # read raw graph from OSM file
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))  # remove isolated nodes .
        self.graph = ox.add_edge_bearings(self.graph)  # add bearing to all edges.
        _, edges = ox.utils_graph.graph_to_gdfs(self.graph)
        # counter with maximum edge id for creating new edges.
        self.edge_osmid = edges['osmid'].max()
        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)
        self.node_type = ['aeroway', 'ref']
        self.save = save
        self.show = show

    def classify_node(self, node_id: int, node_type: str):
        """
        Classified specified node into class.
            - node_id: id of node to classify
            - node_type: string specifying the class into which to classify the node.
        """
        if (node_id in self.graph._node.keys()):
            if (self.graph._node[node_id]['node_type'] == 0 or
               node_type == 'thr_id'):
                self.graph._node[node_id]['node_type'] = self.class_hash[node_type]

    def connect_nodes(self, u: int, v: int, make_bidirectional: bool = True):
        """
        Create edge between specified nodes.
            - u: origin node.
            - v: destination
        """
        self.edge_osmid += 1
        if (self.edge_osmid == 0):
            self.edge_osmid += 1
        self.graph.add_edge(u, v, osmid=self.edge_osmid)
        if (make_bidirectional):
            self.graph.add_edge(v, u, osmid=self.edge_osmid)

    def get_node_colors(self) -> pd.Series:
        """
        Return a series with the corresponding color for each node in the directed graph,
        used for ploting.
        """
        nc = []
        for key in self.graph._node.keys():
            try:
                nc.append(COLOR_CODES[self.graph._node[key]['node_type']])
            except:
                nc.append(COLOR_CODES[0])
        nc = pd.Series(nc, index=self.graph._node.keys())
        return nc

    def fit_edge(self, path, node_separation=30):
        id = max(list(self.graph._node.keys())) + 1
        # Get the coordinates of the start and end nodes
        start_node = path[0]
        end_node = path[-1]
        # Get previous path attributes
        all_nodes_to_remove = []
        # Check if it is a valid edge to super-sample
        if (not start_node in self.graph._node.keys() or not end_node in self.graph._node.keys()):
            return

        all_nodes_to_remove.extend(path[1:-1])
        # Determine node type and coordinates
        node_type = self.graph._node[start_node]['node_type']

        if node_type == self.class_hash['hold_line'] or \
                self.graph._node[end_node]['node_type'] == self.class_hash['hold_line']:
            node_type = self.class_hash['taxiway']

        start_coords = self.graph._node[start_node]['y'], self.graph._node[start_node]['x']
        end_coords = self.graph._node[end_node]['y'], self.graph._node[end_node]['x']
        self.graph.remove_nodes_from(set(all_nodes_to_remove))
        prev_id = start_node
        # Get the total distance of the path
        result = GEOD.Inverse(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        total_distance = result['s12']
        # Calculate the number of intermediate points to add.
        num_points = math.floor(total_distance / node_separation)
        # skip if path is too short to be simplified
        if node_separation > math.floor(total_distance):
            self.connect_nodes(prev_id, end_node, make_bidirectional=False)
            return
        for i in range(1, num_points+1):
            fraction = i / (num_points + 1)
            parametric_distance = fraction * total_distance

            # Use parametric distance to calculate intermediate coordinates
            intermediate_lat = start_coords[0] + \
                (parametric_distance / total_distance) * (end_coords[0] - start_coords[0])

            intermediate_lon = start_coords[1] + \
                (parametric_distance / total_distance) * (end_coords[1] - start_coords[1])

            id += 1
            self.graph.add_node(id)
            self.graph._node[id] = {
                'y': intermediate_lat,
                'x': intermediate_lon,
                'node_type': node_type
            }
            self.connect_nodes(prev_id, id, make_bidirectional=False)
            prev_id = id
        self.connect_nodes(prev_id, end_node, make_bidirectional=False)

        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def semantify_nodes(self):
        """
        Reads the map OSM file and imbues the semantic information of the nodes in the apropiate
        categorical class.

        The information found in the nodes allows for categorizing into:
            - hold_line
        """
        data = ET.parse(self.map_dir)  # Read the data of the OSM file using ET
        root = data.getroot()
        # Iterate through the nodes and assign semantic value to nodes labeled exit,
        # hold_line, thr_id
        for node in root.iter('node'):
            node_id = int(node.attrib['id'])
            if node_id in self.graph._node.keys():
                for tag in node:
                    values = dict(tag.items())
                    key = values['k']
                    if key in self.node_type:
                        zone_type = values['v']
                        if (zone_type == 'holding_position'):
                            self.graph._node[node_id]['zone'] = zone_type
                            self.graph._node[node_id]['node_type'] = self.class_hash['hold_line']
                        else:
                            self.graph._node[node_id]['node_type'] = 0
                if ('node_type' not in self.graph._node[node_id].keys()):
                    self.graph._node[node_id]['node_type'] = 0

    def get_routing_graph(self):
        """
        Further categorized the nodes based on the information stored in the edges into the
        following classes:
            - Bounding: pavement areas or pavement markings.
            - runway (thr_id).
            - taxiway
        Eliminates all nodes and edges that do not correspond the centerline of a movement area.
        """
        nodes_to_remove = []
        data = ET.parse(self.map_dir)
        root = data.getroot()
        for edge in root.iter('way'):
            for tag in edge.findall('tag'):
                values = dict(tag.items())
                if values['k'] == 'ref':
                    zone_type = values['v']
                    # If the zone ID is a letter, it corresponds to a Taxiway.
                    if (zone_type.isalpha()):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id, 'taxiway')

                if values['k'] == 'aeroway':
                    zone_type = values['v']
                    # If the zone ID is a letter, it corresponds to a Taxiway.
                    if (zone_type == 'taxiway'):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id, 'taxiway')

                    if (zone_type == 'runway'):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id, 'thr_id')

        for way in root.findall('way'):
            # Check if this way is a runway
            is_runway = any(tag.get('k') == 'aeroway' and tag.get('v') == 'runway'
                            for tag in way.findall('tag'))

            if is_runway:
                nodes = way.findall('nd')
                if nodes:
                    first_node = nodes[0].get('ref')
                    self.runway_pairs.append(int(first_node))
                    last_node = nodes[-1].get('ref')
                    self.runway_pairs.append(int(last_node))

        self.graph.remove_nodes_from(nodes_to_remove)  # remove the nodes classified as bounding.

    def sanitize(self):
        """
        Sanitized the graph by removing all unclassified nodes and isolated nodes
        left over from previous filtering, and only keeping the biggest subgraph,
        which corresponds to the routing graph of the movement areas.
        Eliminating subgraphs from taxibays, aprons, etc.
        """
        nodes_to_remove = []
        for key in self.graph._node.keys():
            curr_zone = self.graph._node[key]['node_type']
            if curr_zone == 0:
                nodes_to_remove.append(key)
        self.graph.remove_nodes_from(nodes_to_remove)
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        G = (self.graph.subgraph(c) for c in nx.strongly_connected_components(self.graph))
        G = max(G, key=len)
        self.graph = G.copy()
        ox.distance.add_edge_lengths(self.graph)

    def extend_runways(self, extension_distance=1609):
        for i in range(0, len(self.runway_pairs)-1, 2):
            u, v = self.runway_pairs[i], self.runway_pairs[i+1]
            start_point = (self.graph._node[u]['y'], self.graph._node[u]['x'])
            end_point = (self.graph._node[v]['y'], self.graph._node[v]['x'])
            angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

            new_lat_u, new_lon_u = get_new_endpoint(start_point, end_point, extension_distance)
            self.graph._node[u]['x'] = new_lon_u
            self.graph._node[u]['y'] = new_lat_u

            new_lat_v, new_lon_v = get_new_endpoint(end_point, start_point, extension_distance)
            self.graph._node[v]['x'] = new_lon_v
            self.graph._node[v]['y'] = new_lat_v

        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def display_and_save(self, save: bool, show: bool) -> None:
        """
        Plots the graph, if specified also saves in OSM-XML format.
        """
        nc = self.get_node_colors()  # Node colors
        fig, ax = plt.subplots(dpi=1200)
        ox.plot_graph(self.graph, ax=ax, node_size=0.5, node_color=nc, bgcolor="w",
                      edge_color='black', edge_linewidth=0.3, edge_alpha=0.25, show=False)
        if save:
            ox.settings.all_oneway = True
            G = self.graph.copy()
            G = G.to_undirected()
            G_data = ox.graph_to_gdfs(G, nodes=True, edges=True,
                                      node_geometry=True, fill_edge_geometry=True)

            ox.save_graph_xml(G_data,
                              filepath=f"{self.output_dir}/semantic_{self.airport}.osm",
                              node_tags=['x', 'y', 'x_coord', 'y_coord', 'node_type'])

            print(f"Saving to {self.output_dir}/semantic_{self.airport}.osm")
            # Correct way ID #NOTE: Necessary due to the bug in save_graph_xml
            correct_id(f"{self.output_dir}/semantic_{self.airport}.osm", id=self.edge_osmid+1)
            # save png figure
            fig.savefig(self.output_dir+'/sematic_graph.png', dpi=1200, bbox_inches='tight')

        if show:
            plt.show()

    def get_aiport_stats(self):
        G = self.graph.copy()
        Gu = G.to_undirected()
        stats = {}
        # Get Stats
        stats["airport"] = self.airport
        stats["node_count"] = len(G.nodes)
        stats["edge_count"] = len(G.edges)
        stats["average_degree"] = 2 * stats["edge_count"] / stats["node_count"]
        stats["edge_length_total"] = edge_length_total_geod(G)
        stats["edge_length_avg"] = stats["edge_length_total"] / stats["edge_count"]
        stats["self_loop_proportion"] = ox.stats.self_loop_proportion(Gu)
        return stats

    def supersample_graph(self, thr: int = 100, node_separation: int = 30):
        # Convert graph to undirected to avoid supersample in both directions
        G = self.graph.copy()
        G = G.to_undirected()
        self.graph = G
        edges_to_fit = []
        # Iterate through edge dictionary and find edges to supersample
        for u, v, info in self.graph.edges(data=True):
            edge = self.graph[u][v][0]
            start_coords = self.graph._node[u]['y'], self.graph._node[u]['x']
            end_coords = self.graph._node[v]['y'], self.graph._node[v]['x']
            # Calculate edge length in meters (m) with geode for more precise meassurment
            result = GEOD.Inverse(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
            distance = result['s12']
            if (distance > thr):
                edges_to_fit.append([u, v])
        for edge in edges_to_fit:
            u, v = edge[0], edge[1]
            self.graph.remove_edge(u, v)
            self.fit_edge(edge, node_separation=node_separation)
        # Convert back to directed after supersample
        self.graph = self.graph.to_directed()
        ox.distance.add_edge_lengths(self.graph)

    def preprocess_map(self, extension_distance: int = 1609, supersample_value: int = 80,
                       save=True, show=False, verbose=True):
        """
        Defined routine for simplifying OSM map.
        """

        print('Parsing Semantic Information')
        self.semantify_nodes()
        self.get_routing_graph()
        self.extend_runways(extension_distance=extension_distance)
        self.sanitize()
        self.supersample_graph(thr=supersample_value, node_separation=supersample_value)
        self.display_and_save(save, show)

        if (verbose):
            stats = self.get_aiport_stats()
            print_stats(stats=stats)

    def map_to_polylines(self, make_undirected=False):
        if (make_undirected):
            print('Converting undirected graph')
            self.graph = self.graph.to_undirected()

        print('Converting to polyline representation')
        osm_id = 0
        semantic_ids = np.zeros(len(list(self.class_hash.keys())))
        semantic_attribute = 0
        polylines = []
        # zones = []
        for u, v, info in self.graph.edges(data=True):
            ds = [self.graph._node[u]['y'], self.graph._node[u]['x']]
            de = [self.graph._node[v]['y'], self.graph._node[v]['x']]
            ds_xy = calculate_x_y(self.geodesic, self.reference_point, ds)
            de_xy = calculate_x_y(self.geodesic, self.reference_point, de)

            info['osmid'] = osm_id

            if (self.graph._node[u]['node_type'] == self.class_hash['hold_line']):
                self.scenario['hold_lines'].append([
                    ds[0], ds[1],
                    ds_xy[0], ds_xy[1],
                    self.graph._node[u]['node_type'],
                    u
                ])

            if (self.graph._node[v]['node_type'] == self.class_hash['hold_line']):
                self.scenario['hold_lines'].append([
                    de[0], de[1],
                    de_xy[0], de_xy[1],
                    self.graph._node[v]['node_type'],
                    v
                ])

            if ((self.graph._node[u]['node_type'] == self.graph._node[v]['node_type'])):
                semantic_attribute = self.graph._node[u]['node_type']
                self.scenario['map_infos'][self.inv_class_hash[semantic_attribute]].append({
                    'id': semantic_ids[semantic_attribute],
                    'type': semantic_attribute,
                    'polyline_index': (osm_id)
                })
                # zone = self.graph._node[u]['zone']
                semantic_ids[semantic_attribute] += 1

            elif (self.graph._node[u]['node_type'] != self.graph._node[v]['node_type']):

                if (self.graph._node[u]['node_type'] == self.class_hash['thr_id']):
                    semantic_attribute = self.graph._node[v]['node_type']
                else:
                    semantic_attribute = self.graph._node[u]['node_type']

                if self.graph._node[u]['node_type'] == self.class_hash['hold_line'] or \
                        self.graph._node[v]['node_type'] == self.class_hash['hold_line']:
                    semantic_attribute = self.class_hash['hold_line']

                if (self.graph._node[u]['node_type'] == self.class_hash['exit'] or
                   self.graph._node[v]['node_type'] == self.class_hash['exit']):
                    semantic_attribute = self.class_hash['exit']

                self.scenario['map_infos'][self.inv_class_hash[semantic_attribute]].append({
                    'id': semantic_ids[semantic_attribute],
                    'type': semantic_attribute,
                    'polyline_index': (osm_id)
                })
                semantic_ids[semantic_attribute] += 1
            poly_vector = np.array([ds[0], ds[1], ds_xy[0], ds_xy[1],
                                    de[0], de[1], de_xy[0], de_xy[1],
                                    semantic_attribute, osm_id])
            polylines.append(poly_vector)
            # zones.append(zone)
            osm_id += 1
        self.scenario['map_infos']['all_polylines'] = np.asarray(polylines, dtype=np.float32)
        self.scenario['hold_lines'] = np.asarray(self.scenario['hold_lines'], dtype=np.float32)
        # self.scenario['map_infos']['zones'] = zones
        self.scenario['graph_networkx'] = self.graph
        if self.save:
            print(f"Writing pkl to {self.output_dir}/semantic_graph.pkl")
            with open(f"{self.output_dir}/semantic_graph.pkl", 'wb') as handle:
                pkl.dump(self.scenario, handle, protocol=pkl.HIGHEST_PROTOCOL)
        assert self.graph.number_of_edges() == len(polylines)
        print(f"Created polyline with size {len(polylines)}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default=f'{ROOT_DIR}/datasets/amelia', type=str, help='Input path')
    parser.add_argument('--output_dir', default=f'{ROOT_DIR}/output', type=str, help='Output path')
    parser.add_argument('--airport', default='kbos', type=str, help='Airport to process')
    parser.add_argument('--save', action='store_true', default=True, help='Save map')
    parser.add_argument('--show', action='store_true', default=False, help='Show map')
    args = parser.parse_args()
    processor = MapProcessor(**vars(args))
    processor.preprocess_map(save=args.save, show=args.show)
    processor.map_to_polylines()
