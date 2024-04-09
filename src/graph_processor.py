# Configure the directory for necessary imports
import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)

import osmnx as ox
import networkx as nx
import xml.etree.ElementTree as ET 
from src.graph_processor_raw import MapProcessor
from utils.graph_utils import (print_stats, GEOD)

class MapFromNet(MapProcessor):
    def __init__(self, ipath: str, opath: str, 
                 airport: str, load_simplified: bool = False):
        super().__init__(ipath, opath, airport, load_simplified)
        self.node_type = ['aeroway','ref']

    def semantify_nodes(self):
        """
        Reads the map OSM file and imbues the semantic information of the nodes in the apropiate 
        categorical class.

        The information found in the nodes allows for categorizing into:
            - hold_line  
        """ 
        data = ET.parse(self.MAP_DIR) #Read the data of the OSM file using ET
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
                        if(zone_type == 'holding_position'):
                            self.graph._node[node_id]['zone'] = zone_type
                            self.graph._node[node_id]['node_type'] = self.class_hash['hold_line']
                        else:
                            self.graph._node[node_id]['node_type'] = 0
                if('node_type' not in self.graph._node[node_id].keys()):
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
        data = ET.parse(self.MAP_DIR)
        root = data.getroot()
        for edge in root.iter('way'):
            for tag in edge.findall('tag'):
                values = dict(tag.items())
                if values['k'] == 'ref':
                    zone_type = values ['v']
                    # If the zone ID is a letter, it corresponds to a Taxiway.
                    if(zone_type.isalpha()):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'taxiway')

                if values['k'] == 'aeroway':
                    zone_type = values ['v']
                    # If the zone ID is a letter, it corresponds to a Taxiway.
                    if(zone_type == 'taxiway'):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'taxiway')

                    if(zone_type == 'runway'):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'thr_id')


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

        self.graph.remove_nodes_from(nodes_to_remove) # remove the nodes classified as bounding.

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

    def supersample_graph(self, thr: int = 100, node_separation: int = 30):
        # Convert graph to undirected to avoid supersample in both directions
        G = self.graph.copy()
        G = G.to_undirected()
        self.graph = G
        edges_to_fit = []
        # Iterate through edge dictionary and find edges to supersample
        for u, v, info in self.graph.edges(data=True):
            edge   = self.graph[u][v][0]
            start_coords = self.graph._node[u]['y'], self.graph._node[u]['x']
            end_coords = self.graph._node[v]['y'], self.graph._node[v]['x']
            # Calculate edge length in meters (m) with geode for more precise meassurment
            result = GEOD.Inverse(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
            distance = result['s12']
            if(distance > thr):
                edges_to_fit.append([u,v])
        for edge in edges_to_fit:
            u, v = edge[0], edge[1]
            self.graph.remove_edge(u,v)
            self.fit_edge(edge, node_separation= node_separation)
        # Convert back to directed after supersample
        self.graph = self.graph.to_directed()
        ox.distance.add_edge_lengths(self.graph)  

    def preprocess_map(self, extension_distance: int = 1609, supersample_value: int = 80, 
                       save = True, show = False, verbose = True):
        """
        Defined routine for simplifying OSM map.
        """
        if(self.pre_simplified): 
            print('Loading presimplified graph, skipping simplification')
            self.display_and_save(save= save)
            return
        print('Parsing Semantic Information')
        self.semantify_nodes()
        self.get_routing_graph()
        self.extend_runways(extension_distance= extension_distance)
        self.sanitize()
        self.supersample_graph(thr = supersample_value, node_separation = supersample_value)
        if(show):
            self.display_and_save(save)

        if(verbose):
            stats = self.get_aiport_stats()
            print_stats(stats = stats)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default = '../boeing/swim/', type=str, help='Input path.')
    parser.add_argument('--opath', default = 'out', type=str, help='Output path.')
    parser.add_argument('--airport', default= 'kewr', type=str, help='Airport to process.')
    parser.add_argument('--load_simplified',  
                        help='Determines if the processed graph has been previously simplified',
                        action='store_true')
    args = parser.parse_args()
    processor = MapFromNet(**vars(args))
    processor.preprocess_map(save= True, show = True)
    processor.map_to_polylines()