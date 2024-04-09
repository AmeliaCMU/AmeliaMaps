import os
import json
import osmnx as ox
import networkx as nx
import xml.etree.ElementTree as ET 
import math
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from geographiclib.geodesic import Geodesic
import numpy as np
from utils.graph_utils import (_is_endpoint, _get_paths_to_simplify, calculate_x_y, 
                         COLOR_CODES, get_path_attributes, get_new_endpoint, correct_id, GEOD, edge_length_total_geod)


class MapProcessor():
    """
    A class for proccessing airport routing graph in OSM format. And generating a pickled dictionary
    containing the edges of the graph similar to VectorNet's polyline representation. https://arxiv.org/abs/2005.04259
        - ipath: Path where to access the osm files.
        - opath: Path to store the generated visualizations of the graph.
        - airport: The name of the airport or aiport file name.
    """
    def __init__(self, ipath: str, opath: str, 
                 airport: str, load_simplified: bool = False):
        # Specify paths for necessary files and output directory.
        self.AIRPORT = airport
        self.BASE_DIR = ipath
        self.OUT_DIR = os.path.join(opath,self.AIRPORT)
        if not os.path.exists(self.OUT_DIR):
            os.makedirs(self.OUT_DIR)

        self.MAP_DIR = os.path.join(self.BASE_DIR, 'maps', self.AIRPORT, f'{self.AIRPORT}_from_net.osm')
        LIMITS_FILE = os.path.join(self.BASE_DIR,'maps', self.AIRPORT, 'limits.json')
        self.pre_simplified = load_simplified
        self.geodesic = Geodesic.WGS84 #geoide of the Earth
        with open(LIMITS_FILE, 'r') as f:
            self.ref_data = json.load(f)
        self.reference_point = (self.ref_data['ref_lon'],self.ref_data['ref_lat'])
        node_file = open("utils/node_types.txt", "r") # process the abstract node classes to classify nodes.
        self.node_type = node_file.read().split(',')
        # Create dictionaries used for processing.
        map_info = {}
        map_info['all_polylines'] = np.array([])
        self.class_hash = {} # Hash to convert from string value to numerical value.
        self.inv_class_hash = {} # Hash to convert from numerical value to string.
        self.scenario = {}
        self.scenario['hold_lines'] = []
        # Populate dictionaries based on the reported node types
        for i in range(0,len(self.node_type)):
            val = i
            if(i!=0):
                map_info[self.node_type[i]] = []
                self.inv_class_hash[val] = self.node_type[i]
            self.class_hash[self.node_type[i]] = val
        self.scenario['map_infos'] = map_info
        self.tags = ['x','y']
        self.runway_pairs = []
        # If loading from a previosly simplified graph specify tags to load
        if(self.pre_simplified):
            tags = ['node_type', 'osmid']
            ox.settings.useful_tags_node += tags
        # Load graph from XML file, and do preliminary clean-up.
        self.graph = ox.graph_from_xml(self.MAP_DIR, retain_all=True, simplify=False) # read raw graph from OSM file
        self.graph.remove_nodes_from(list(nx.isolates(self.graph))) # remove isolated nodes .
        self.graph = ox.add_edge_bearings(self.graph) # add bearing to all edges.
        _, edges = ox.utils_graph.graph_to_gdfs(self.graph)
        self.edge_osmid = edges['osmid'].max() # counter with maximum edge id for creating new edges.
        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)
        # If loading from a previosly simplified graph, correct semantic categorical value
        if(self.pre_simplified):
            print('Loading Simplified Graph')
            for key in self.graph._node.keys():  
                try: self.graph._node[key]['node_type'] = int(self.graph._node[key]['node_type'])
                except: print(f"Node with key {key} does not have node_type")

    def classify_node(self, node_id: int, node_type :str):
        """
        Classified specified node into class.
            - node_id: id of node to classify 
            - node_type: string specifying the class into which to classify the node.
        """
        if(node_id in self.graph._node.keys()):
            if(self.graph._node[node_id]['node_type'] == 0 or 
               node_type == 'thr_id'):
                self.graph._node[node_id]['node_type'] = self.class_hash[node_type]
    
    def connect_nodes(self, u: int,v:int, make_bidirectional: bool = True):
        """
        Create edge between specified nodes.
            - u: origin node.
            - v: destination
        """
        self.edge_osmid += 1
        if(self.edge_osmid == 0):
            self.edge_osmid +=1
        self.graph.add_edge(u,v,osmid = self.edge_osmid)
        if(make_bidirectional):
            self.graph.add_edge(v,u, osmid = self.edge_osmid)
        
    def get_node_colors(self) -> pd.Series:
        """
        Return a series with the corresponding color for each node in the directed graph, 
        used for ploting.
        """
        nc = []
        for key in self.graph._node.keys():
            try: nc.append(COLOR_CODES[self.graph._node[key]['node_type']])
            except: nc.append(COLOR_CODES[0])
        nc = pd.Series(nc,index=self.graph._node.keys())
        return nc

    def simplify_path(self, path: list, print_merged: bool = False):
        """
        Receives a list of node IDs. It combines said nodes into a single edge, 
        taking as reference the edge of the first node.
            - path: list of nodes to simplify
            - print_merged: flag to toggle printing of merged node IDs. 
        
        """
        all_nodes_to_remove = []
        all_edges_to_add = []
        path_attributes, merged_edges = get_path_attributes(path, self.graph)
        if(path_attributes == None or merged_edges == None): return
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
        )
        for edge in all_edges_to_add:
            self.edge_osmid+=1
            self.graph.add_edge(edge["destination"], edge["origin"],
                                 **{**edge["attr_dict"],'osmid': self.edge_osmid})
            self.edge_osmid+=1
            self.graph.add_edge(edge["origin"], edge["destination"],
                                 **{**edge["attr_dict"],'osmid': self.edge_osmid})
        if print_merged:print(merged_edges)
        self.graph.remove_nodes_from(set(all_nodes_to_remove))
        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def fit_edge(self, path, node_separation = 30):
        id = max(list(self.graph._node.keys())) + 1
        # Get the coordinates of the start and end nodes
        start_node = path[0]
        end_node = path[-1]
        # Get previous path attributes
        all_nodes_to_remove = []
        # Check if it is a valid edge to super-sample
        if(not start_node in self.graph._node.keys() or not end_node in self.graph._node.keys()):
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
            self.connect_nodes(prev_id, end_node, make_bidirectional = False)
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
            self.connect_nodes(prev_id, id, make_bidirectional = False)
            prev_id = id
        self.connect_nodes(prev_id, end_node, make_bidirectional = False)

        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def semantify_nodes(self):
        """
        Reads the map OSM file and imbues the semantic information of the nodes in the apropiate 
        categorical class.

        The information found in the nodes allows for categorizing into:
            - hold_line
            - exit  
        """ 
        data = ET.parse(self.MAP_DIR) #Read the da of the OSM file using ET
        root = data.getroot()
        # Iterate through the nodes and assign semantic value to nodes labeled exit, 
        # hold_line, thr_id
        for node in root.iter('node'):
            node_id = int(node.attrib['id'])
            if node_id in self.graph._node.keys():
                for tag in node:
                    values = dict(tag.items())
                    key = values['k']
                    # if key in ['x', 'y'] and key not in self.graph:
                    #     self.graph._node[node_id][key+'_coord'] = float(values['v'])
                    if key in self.node_type:
                        value = self.class_hash[key]
                        self.graph._node[node_id]['node_type'] = value
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
                if values['k'] == 'rwy_id':
                    # Get the node items that correspond to pavement
                    for node in edge.findall('nd'): 
                        node_id = int(node.attrib['ref'])
                        # Reading the ref tag value INFO of the zone
                        self.graph._node[node_id]['zone'] = values['v'] 
                        self.runway_pairs.append(node_id)

                # If the edge contains the attribute area with value true, it corresponds 
                # to a bounding zone.
                if values['k'] == 'area':
                    zone_type = values ['v']
                    #Find nodes that correspond to pavement areas.
                    if(zone_type == 'true'): 
                        # Get the node items that correspond to pavement
                        for node in edge.findall('nd'): 
                            node_id = int(node.attrib['ref'])
                            if(node_id in self.graph._node.keys()):
                                if(self.graph._node[node_id]['node_type'] == 0):
                                    # If the node is not essential, classify as node to remove them.
                                    self.graph._node[node_id]['node_type'] = \
                                        self.class_hash['bounding'] 
                                    nodes_to_remove.append(node_id)

                # The value in zone_id allows us to further classify the nodes in the edge.
                if values['k'] == 'zone_id':
                    zone_type = values ['v']
                    # If the zone ID is a letter, it corresponds to a Taxiway.
                    if(zone_type.isalpha()):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'taxiway')
                    # Classify ramps
                    elif('Ramp' in zone_type):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'ramp')
                    elif(not '$UNK' in zone_type and not 'apron' in zone_type.lower()):
                        for node in edge.findall('nd'):
                            node_id = int(node.attrib['ref'])
                            self.graph._node[node_id]['zone'] = zone_type
                            self.classify_node(node_id,'taxiway')

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
        self.graph = ox.simplify_graph(self.graph, False, True, True)
        ox.distance.add_edge_lengths(self.graph)

    def subsample_roadgraph(self, zone: str = "taxiway", path_depth: int = 6, strategy = 'remove'):
        """ 
        For the specified zone, identifies endpoints and travers between those consolidating every 
        certain nodes.
            - zone: string with node type to simplify.
            - path_depth: length of simplification per path. 
        """
        nodes_in_zone = [ n for n in list(self.graph._node.keys()) 
                            if self.graph._node[n]['node_type'] == self.class_hash[zone]]
        endpoints = set([n for n in nodes_in_zone if _is_endpoint(self.graph, n, strict=True)])
        zones_to_avoid = [self.class_hash['hold_line'], self.class_hash['exit']]
        paths = list(_get_paths_to_simplify(endpoints, self.graph, zones_to_avoid))
        for path in paths:
            while(path):
                subpath, path = path[:path_depth], path[(path_depth-1):]
                if(len(path) > 2):
                    if(strategy == 'remove'): 
                        self.simplify_path(subpath)
                    elif(strategy == 'interpolation'):
                        self.fit_edge(path)

        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)
                                      
    def subsample_roadgraph_iterative(self, zone: str = "taxiway", iterations:str = 10, 
                                      path_depth:str = 6):
        """
        Iteratively consolidates nodes between endpoints, perfoming one consolidation per iteration.
            - zone: string with node type to simplify.
            - iterations: number of times to perform simplifications.
            - path_depth: length of simplification per iteration.
        """
        nodes_in_zone = [ n for n in list(self.graph._node.keys()) 
                            if self.graph._node[n]['node_type'] == self.class_hash[zone]]
        # source = [random.choice(nodes_in_zone)]
        endpoints = set([n for n in nodes_in_zone if _is_endpoint(self.graph, n, strict=True)])
        for i in range(iterations):
            # print(f"Identified {len(endpoints)} edge endpoints")
            for source in endpoints:
                path = []
                counter = 0
                if len(list(self.graph.successors(source))) > 0:
                    next_node = list(self.graph.successors(source))[0]
                    path.append(source)
                    path.append(next_node)
                    if next_node in endpoints:
                        continue
                    for successor in self.graph.successors(next_node):
                        if successor not in path:
                            path.append(successor)
                            while (counter < (path_depth-2) and (successor not in endpoints) and
                                    self.graph._node[successor]['node_type'] != \
                                        self.class_hash['hold_line']):
                                successors = [n for n in \
                                              self.graph.successors(successor) if n not in path]
                                if len(successors) == 1:
                                    successor = successors[0]
                                    path.append(successor)
                                elif len(successors) == 0:
                                    break
                                counter+=1
                    self.simplify_path(path)

    def extend_runways(self, extension_distance = 1609):
        for i in range(0,len(self.runway_pairs)-1,2):
            u, v = self.runway_pairs[i], self.runway_pairs[i+1]
            start_point = (self.graph._node[u]['y'],self.graph._node[u]['x'])
            end_point = (self.graph._node[v]['y'],self.graph._node[v]['x'])
            angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
           
            new_lat_u, new_lon_u = get_new_endpoint(start_point, end_point, extension_distance)
            self.graph._node[u]['x'] = new_lon_u
            self.graph._node[u]['y'] = new_lat_u

            new_lat_v, new_lon_v = get_new_endpoint(end_point, start_point, extension_distance)
            self.graph._node[v]['x'] = new_lon_v
            self.graph._node[v]['y'] = new_lat_v

        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def pad_centerline(self, node_separation: int = 10):
        id = max(list(self.graph._node.keys())) + 1
        line_fit = []
        for i in range(0,len(self.runway_pairs)-1,2):
            u, v = self.runway_pairs[i], self.runway_pairs[i+1]
            length = self.graph[u][v][0]['length']
            cords = [
                (self.graph._node[u]['x'],self.graph._node[u]['y']),
                (self.graph._node[v]['x'],self.graph._node[v]['y'])]
            slope = (cords[1][1] - cords[0][1])/(cords[1][0] - cords[0][0])
            intercept = cords[1][1] - (slope*cords[1][0])
            line_fit.append((cords[0][0], cords[1][0], slope,intercept, u, v, length ))
            self.graph.remove_edge(self.runway_pairs[i], self.runway_pairs[i+1])
            self.graph.remove_edge(self.runway_pairs[i+1], self.runway_pairs[i])

        for fit in line_fit:
            start, stop, slope, intercept, start_id, stop_id, length = fit
            n_points = int(length // node_separation)
            step = (stop - start)/n_points
            prev_id = start_id
            for point in range(n_points):
                id += 1
                start += step
                self.graph.add_node(id)
                self.graph._node[id] = {
                    'y': slope * start + intercept,
                    'x': start,
                    'node_type': self.class_hash['thr_id']
                }
                self.connect_nodes(prev_id, id, make_bidirectional= True)
                prev_id = id
            self.connect_nodes(prev_id, stop_id, make_bidirectional= True)
        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def connect_to_runway(self, min_distance: float = 0.004):
        """
        Iterates through the exit nodes and connects them to the nearest runway nodes.
            - min_distance: distance threshold to make a connection
        """
        runway = []
        nodes_to_connect = []
        dist = 1
        for key in self.graph._node.keys():
            curr_zone = self.graph._node[key]['node_type']
            if curr_zone == self.class_hash['thr_id']:
                runway.append(key)
            if curr_zone == self.class_hash['exit']:
                nodes_to_connect.append(key)
        for u in nodes_to_connect:
            dist = 1
            node_to_connect = u
            for v in runway:
                euclid = math.dist([self.graph._node[u]['x'],self.graph._node[u]['y']],
                                   [self.graph._node[v]['x'],self.graph._node[v]['y']])
                if abs(euclid) < abs(dist):
                    dist = euclid
                    node_to_connect = v
            if(abs(dist) < min_distance): 
                self.connect_nodes(u, node_to_connect, make_bidirectional=True)
        ox.distance.add_edge_lengths(self.graph)
        ox.add_edge_bearings(self.graph)

    def display_and_save(self, save):
        """
        Plots the graph, if specified also saves in OSM-XML format.
        """
        nc = self.get_node_colors() # Node colors
        fig, ax = plt.subplots(dpi=1200) #
        ox.plot_graph(self.graph, ax=ax, node_size = 0.5, node_color=nc, bgcolor="w",
                      edge_color='black',edge_linewidth = 0.3, edge_alpha=0.25)
        if save: 
            ox.settings.all_oneway = True
            G = self.graph.copy()
            G = G.to_undirected()
            G_data = ox.graph_to_gdfs(G, nodes=True, edges=True,
                                        node_geometry=True, fill_edge_geometry=True)
            
            ox.save_graph_xml(G_data,
                               filepath=f"{self.OUT_DIR}/semantic_{self.AIRPORT}.osm",
                               node_tags=['x','y','x_coord','y_coord','node_type'])
            
            print(f"Saving to {self.OUT_DIR}/semantic_{self.AIRPORT}.osm")
            # Correct way ID #NOTE: Necessary due to the bug in save_graph_xml
            correct_id(f"{self.OUT_DIR}/semantic_{self.AIRPORT}.osm", id= self.edge_osmid+1)

        if save: fig.savefig(self.OUT_DIR+'/sematic_graph',dpi=1200,bbox_inches='tight')

    def get_aiport_stats(self):
        G = self.graph.copy()
        Gu = G.to_undirected()
        stats = {}
        # Get Stats
        stats["airport"]    = self.AIRPORT
        stats["node_count"] = len(G.nodes)
        stats["edge_count"] = len(G.edges)
        stats["average_degree"] = 2 * stats["edge_count"] / stats["node_count"]
        stats["edge_length_total"] = edge_length_total_geod(G)
        stats["edge_length_avg"] = stats["edge_length_total"] / stats["edge_count"]
        stats["self_loop_proportion"] = ox.stats.self_loop_proportion(Gu)
        return stats

    def preprocess_map(self):
        """
        Defined routine for simplifying OSM map.
        """
        if(self.pre_simplified): 
            print('Loading presimplified graph, skipping simplification')
            self.display_and_save(False)
            return
        print('Parsing Semantic Information')
        self.semantify_nodes()
        self.get_routing_graph()
        self.pad_centerline(node_separation = 60)
        self.connect_to_runway()
        self.sanitize()
        self.subsample_roadgraph(zone='taxiway',path_depth = 4, strategy='remove')
        print(f"Processed graph with {self.graph.number_of_edges()} edges.")

    def map_to_polylines(self, make_undirected = False):
        if(make_undirected):
            print('Converting undirected graph')
            self.graph = self.graph.to_undirected()

        print('Converting to polyline representation')
        osm_id = 0
        semantic_ids = np.zeros(len(list(self.class_hash.keys())))
        semantic_attribute = 0
        polylines = []
        zones = []
        for u, v, info in self.graph.edges(data=True):
            ds     = [self.graph._node[u]['y'],self.graph._node[u]['x']]
            de     = [self.graph._node[v]['y'],self.graph._node[v]['x']]
            ds_xy  = calculate_x_y(self.geodesic, self.reference_point,ds)
            de_xy  = calculate_x_y(self.geodesic, self.reference_point,de)
            
            info['osmid'] = osm_id

            if(self.graph._node[u]['node_type'] == self.class_hash['hold_line']):
                self.scenario['hold_lines'].append([
                    ds[0], ds[1],
                    ds_xy[0], ds_xy[1],
                    self.graph._node[u]['node_type'], 
                    u
                ])
            
            if(self.graph._node[v]['node_type'] == self.class_hash['hold_line']):
                self.scenario['hold_lines'].append([
                    de[0], de[1],
                    de_xy[0], de_xy[1],
                    self.graph._node[v]['node_type'], 
                    v
                ])

            if((self.graph._node[u]['node_type'] == self.graph._node[v]['node_type'])):
                semantic_attribute = self.graph._node[u]['node_type']
                self.scenario['map_infos'][self.inv_class_hash[semantic_attribute]].append({
                    'id': semantic_ids[semantic_attribute],
                    'type': semantic_attribute,
                    'polyline_index' : (osm_id)
                })
                # zone = self.graph._node[u]['zone']
                semantic_ids[semantic_attribute]+=1

            elif(self.graph._node[u]['node_type'] != self.graph._node[v]['node_type'] ):

                if(self.graph._node[u]['node_type'] == self.class_hash['thr_id']):
                    semantic_attribute = self.graph._node[v]['node_type']
                else:
                    semantic_attribute = self.graph._node[u]['node_type']

                if self.graph._node[u]['node_type'] == self.class_hash['hold_line'] or \
                    self.graph._node[v]['node_type'] == self.class_hash['hold_line']: 
                    semantic_attribute = self.class_hash['hold_line']
            
                if(self.graph._node[u]['node_type'] == self.class_hash['exit'] or \
                   self.graph._node[v]['node_type'] == self.class_hash['exit']):
                    semantic_attribute = self.class_hash['exit'] 

                self.scenario['map_infos'][self.inv_class_hash[semantic_attribute]].append({
                    'id': semantic_ids[semantic_attribute],
                    'type': semantic_attribute,
                    'polyline_index' : (osm_id)
                })
                semantic_ids[semantic_attribute]+=1
            poly_vector = np.array([ds[0],ds[1],ds_xy[0],ds_xy[1],
                                    de[0],de[1],de_xy[0],de_xy[1], semantic_attribute, 
                                    osm_id])
            polylines.append(poly_vector)
            # zones.append(zone)
            osm_id+=1
        self.scenario['map_infos']['all_polylines'] = np.asarray(polylines,dtype=np.float32)
        self.scenario['hold_lines']= np.asarray(self.scenario['hold_lines'],dtype=np.float32 )
        # self.scenario['map_infos']['zones'] = zones
        self.scenario['graph_networkx'] = self.graph
        print(f"Writing pkl to {self.OUT_DIR}/semantic_graph.pkl")
        with open(f"{self.OUT_DIR}/semantic_graph.pkl", 'wb') as handle:
            pkl.dump(self.scenario, handle, protocol=pkl.HIGHEST_PROTOCOL)
        assert self.graph.number_of_edges() == len(polylines)
        print(f"Created polyline with size {len(polylines)}")
            
