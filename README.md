# Graph Processing


## Overview


Tools for creating vectorized context graphs for intent prediction. The graph processors assume the maps are in a folder with the following structure.


```
├── BASE_DIR
│   ├── maps
│   │   ├── airport_code
|   |   |    ├── airport_code_from_net.osm
|   |   |    ├── limits.json         
```


After simplifying the desired osm file manually, use the graph_processor.py tool to generate the corresponding pickle file containing the polyline dictionary for the graph. Run the following command from the graph_construction.


```
cd swim_data_tools
cd ./graph_construction
```
```
python src/graph_processor.py  --airport <AIRPORT> --ipath <BASE_DIR> --opath <OUTDIR>
```


Note that the load simplified tag allows for loading manually pre simplified graphs containing the necessary node tags. If this flag is set to true, the preprocessing pipeline will be skipped. To enable this feature, add the load_simplify argument.


For visualizing the simplification process you can use the [semantic_graph notebook](semantic_graph.ipynb) to see the effect of each step in the graph. Currently, the sanitize function only keeps the largest subgraph, so make sure all the relevant movement areas are fully connected.


## Manual pre-processing


When using this tool with airports downloaded from OSM, these need to have certain characteristics to be compatible with the processing script.


- Runways must be grouped in a single edge. There can be multiple nodes on the runway, but these must be grouped in one single way.
- Runway end nodes cannot be connected "laterally" to taxiways or holdlines. This is because the centerlines will be extended. If a ramp connects to the end of the runway, insert a surrogate node below it.
- Edges must contain aeroway keys indicating the type of zone, i.e. `aeroway: taxiway`. Hold lines nodes must have the equivalent tag `aeroway : holding_position`.


## The MapProcessing Class


The core behind the vectorized map processing repo is the Map Processing class found [here](src/graph_processor_raw.py). The base class allows for processing OSM files in a proprietary format. For processing the OSM files obtained from public sources, use `MapFromNet` found [here](src/graph_processor.py).


Each of the steps in the processing pipeline is separated into methods of the class. All these methods affect the graph attribute of the class.


### MapFromNet.preprocess_map(extension_distance = 1609, supersample_value = 80, save = True, show = False, verbose = True)
Contains the predefined routine for simplifying the routing graphs.


__PARAMETERS__
- **_extension_distance_**: Distance in meters to extend runway endpoints. The default is 1,609, equivalent to 1 nautical mile.
- **_supersample_value_**: Value in meters to supply to the supersampling method. Note that the supersampling threshold and node spacing are the same for consistency in the processed graph.
- **_save_**: flag for toggling saving the processed graph in XML format.
- **_show_**: flag for toggling plotting the processed graph.
- **_verbose_**: If set to true, it displays a statistic summary of the processed graph.


### MapFromNet.map_to_polylines()
Once the map has been preprocessed, it can be converted to vectorized form and stored as a pickle file using this method.


### MapFromNet.semantify_nodes()
Given the raw graph, search the nodes for relevant aeroway tags and classify important nodes.


### MapFromNet.get_routing_graph()
Classifies the relevant centerlines for movement areas and removes unnecessary edges. Runway endpoint nodes are stored in the `MapFromNet.runway_pairs` attribute. Consecutive elements in this list represent the pair of endpoints for a given runway.


### MapFromNet.extend_runways(extension_distance)
Displaces all runway endpoints by the specified amount following the slope of the runway.


__PARAMETERS__
- **_extension_distance_**: Distances in meters to extend runway endpoints. The default is 1,609, equivalent to 1 nautical mile


### MapFromNet.sanitize()
Removes all nodes with default class (node_type == 0) and keeps only the largest subgraph. This assumes the movement area of the airport is a fully connected graph.


### MapFromNet.supersample_graph(thr, node_separation)
Identifies edges below the specified threshold and supersamples them, placing a node every specified meters.


__PARAMETERS__
- **_thr_**: Minimum length, in meters, of an edge to be supersampled.
- **_node_separation_**: separation in meters between the placed nodes.


### MapFromNet.display_and_save(save)
Displays the graph found in `MapFromNet.graph` with the semantic colors. This requires the nodes to have a node_type attribute.


__PARAMETERS__
- **_save_**: Flag to save the graph in XML format. The graph is saved under `<OUT_DIR>/<AIRPORT>/semantic_<AIRPORT>.osm`