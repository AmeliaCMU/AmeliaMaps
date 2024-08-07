import xml.etree.ElementTree as ET
import pandas as pd
import networkx
import numpy as np
from typing import Tuple, List
from shapely.geometry import LineString
from shapely.geometry import Point
from geographiclib.geodesic import Geodesic
from math import cos, sin, radians
from tabulate import tabulate
import os


COLOR_CODES = {
    0: '#f05eee',
    1: '#ff9b21',
    2: '#fc0349',
    3: '#1fe984',
    4: '#6285cf',
    5: '#f05eee',
}

GEOD = Geodesic.WGS84
EARTH_RADIUS = 6378137

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def edge_length_total_geod(G):
    """
    Calculate graph's total edge length.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    length : float
        total length (meters) of edges in graph
    """
    # NOTE: this overloads the default OSMNx function and performs the calculation using geods for more presicion.
    return sum(get_edge_length(G, u, v) for u, v, d in G.edges(data=True))


def get_edge_length(G, u, v):
    start_coords = G._node[u]['y'], G._node[u]['x']
    end_coords = G._node[v]['y'], G._node[v]['x']
    # Calculate edge length in meters (m) with geode for more precise meassurment
    result = GEOD.Inverse(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
    distance = result['s12']
    return distance


def print_stats(stats):
    table_data = []
    for key, value in stats.items():
        if "length" in key:
            key = f"{key} (m)"
        table_data.append((f"\033[1m{key}\033[0m", value))
    # Print the formatted table
    print(f"\033[1;35mAirport {stats['airport'].upper()} Summary\033[0m")
    print("-" * 30)
    print(tabulate(table_data, headers=["Statistic", "Value"],
          tablefmt="fancy_grid", colalign=("center", "center")))


def correct_id(file, id):
    # Load the XML file
    tree = ET.parse(file)
    root = tree.getroot()

    # Define the target way ID
    target_way_id = '0'

    # Find the <way> element with the specified ID
    target_way_element = None
    for way_element in root.findall('.//way'):
        way_id = way_element.attrib.get('id')
        if way_id == target_way_id:
            target_way_element = way_element
            break

    # Check if the target way element is found
    if target_way_element is not None:
        # Modify the value of the way element (for example, change the attribute value)
        target_way_element.attrib['id'] = str(id)
        # Save the modified XML to a new file

        tree.write(file, encoding="utf-8", xml_declaration=True)

    else:
        print(f"Way with ID={target_way_id} not found in the XML file.")


def get_new_endpoint(start_coords, end_coords, distance_meters):
 # Create a Geodesic object
    geod = Geodesic.WGS84

    # Calculate the initial bearing from start to end
    line = geod.Inverse(end_coords[0], end_coords[1], start_coords[0], start_coords[1])
    initial_bearing = line['azi1']

    # Calculate the destination point using the initial bearing and distance
    destination = geod.Direct(start_coords[0], start_coords[1], initial_bearing, distance_meters)

    return (destination['lat2'], destination['lon2'])


def plot_hold_line(hold_lines, ax):
    ax.scatter(hold_lines[:, 1], hold_lines[:, 0], c='#ff9b21')


def plot_context(context, ax, xyminmax=None):
    """ Visualization tool to plot the map context.

    Inputs
    ------
        context[np.array]: vectors to plot
        ax[plt.axis]: matplotlib axis to plot on.
    """
    for pl in context:
        # breakpoint()
        if len(pl) == 5:
            y_start_rl, x_start_rl, y_end_rl, x_end_rl, semantic_id = pl  # Pre-onehot encoding
        # Post-onehot encoding
        else:
            x_start_rl = pl[1]
            y_start_rl = pl[0]
            x_end_rl = pl[3]
            y_end_rl = pl[2]

        if x_start_rl == 0 and x_end_rl == 0 and y_start_rl == 0 and y_end_rl == 0:
            continue

        if len(pl) == 5:
            color = COLOR_CODES[semantic_id]
        else:
            color_id = (np.where(pl == 1)[0][0] - 4) + 1
            color = COLOR_CODES[color_id]

        if xyminmax is not None:
            xmin, ymin, xmax, ymax = xyminmax
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_aspect('equal')
        ax.plot([x_start_rl, x_end_rl], [y_start_rl, y_end_rl], color=color, linewidth=1.2, alpha=1)


def get_path_attributes(path, graph):
    # Get previous path attributes
    merged_edges = []
    attrs_to_sum = {"length", "travel_time"}
    path_attributes = {}
    # Iterate through edges to get attributes and copy to new node.
    for u, v in zip(path[:-1], path[1:]):
        merged_edges.append((u, v))
        edge_count = graph.number_of_edges(u, v)
        if edge_count != 1:
            return (None, None)
        edge_data = list(graph.get_edge_data(u, v).values())[0]
        for attr in edge_data:
            if attr in path_attributes:
                path_attributes[attr].append(edge_data[attr])
            else:
                path_attributes[attr] = [edge_data[attr]]
    # Collapse attributes by summing them or by taking the first element of the attribute array
    for attr in path_attributes:
        try:
            if attr in attrs_to_sum:
                path_attributes[attr] = sum(path_attributes[attr])

            elif len(set(path_attributes[attr])) == 1:

                path_attributes[attr] = path_attributes[attr][0]
            else:

                path_attributes[attr] = list(set(path_attributes[attr]))
        except:
            return (None, None)
    path_attributes["geometry"] = LineString(
        [Point((graph.nodes[node]["x"], graph.nodes[node]["y"])) for node in path]
    )
    return (path_attributes, merged_edges)


def get_node_colors(graph: networkx.MultiDiGraph) -> pd.Series:
    """
    Return a series with the corresponding color for each node in the directed graph, used for ploting.
    """
    nc = []
    for key in graph._node.keys():
        try:
            nc.append(COLOR_CODES[graph._node[key]['node_type']])
        except:
            nc.append(COLOR_CODES[0])

    nc = pd.Series(nc, index=graph._node.keys())
    return nc


def get_range_and_bearing(geodesic: Geodesic, reference_point: Tuple,
                          point: Tuple, to_radians: bool = True) -> Tuple:
    """
    Computes the bearing angle and range in meters between the given point and the specified reference.
    ---
    Inputs:
        - geodesic: geode of the the earth.
        - reference_point: tuple containing the point with respect to which calculate bearing and range (lon, lat).
        - point: tuple containing the input point.
        - to_radians: flag that indicates whether the bearing will be reported in radians.
    """
    ref_lon, ref_lat = reference_point
    g = geodesic.Inverse(ref_lat, ref_lon, point[0], point[1])
    bearing = g['azi1']
    range = g['s12']
    range = range/1000
    if to_radians:
        bearing = radians(g['azi1'])
    return (range, bearing)


def calculate_x_y(geodesic: Geodesic, reference_point: Tuple, point: Tuple, to_radians=True):
    """
    Computes the local cartesian cordiantes of a point given a specified reference.
    ---
    Inputs:
        - geodesic: geode of the the earth.
        - reference_point: tuple containing the point with respect to which calculate bearing and range (lon, lat).
        - point: tuple containing the input point.
        - to_radians: flag that indicates whether the bearing will be reported in radians.
    """
    r, b = get_range_and_bearing(geodesic, reference_point, point)
    # Apply conversion formula
    x = r * cos(b)
    y = r * sin(b)
    return [x, y]


def _is_endpoint(G, node, strict=True):
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its List of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    elif not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]["osmid"])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]["osmid"])

        # if there is more than 1 OSM ID in the List of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    # if none of the preceding rules returned true, then it is not an endpoint
    else:
        return False


def _build_path(G, endpoint, endpoint_successor, endpoints, zones_to_avoid):
    """
    Build a path of nodes from one endpoint node to next endpoint node.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    endpoint : int
        the endpoint node from which to start the path
    endpoint_successor : int
        the successor of endpoint through which the path to the next endpoint
        will be built
    endpoints : set
        the set of all nodes in the graph that are endpoints

    Returns
    -------
    path : List
        the first and last items in the resulting path List are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints and G._node[successor]['node_type'] not in zones_to_avoid:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:  # pragma: no cover
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    return []

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path


def _get_paths_to_simplify(endpoints, G, zones_to_avoid):
    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if (successor not in endpoints) and (G._node[successor]['node_type'] not in zones_to_avoid):
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints, zones_to_avoid)
