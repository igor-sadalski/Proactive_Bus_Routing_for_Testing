#!/usr/bin/env python
"""Map_graph.py

This script downloads the street network associated with a given polygon igven as a list of coordinates in a text file. It also calculates pair-wise shortest distance between intersections
if initialize_shortest_path=True.

Example:
    Default usage:
        $ python3 Map_graph.py

"""

import os
import math
import pickle
import traceback
import datetime
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

class Map_graph:
    def __init__(self, initialize_shortest_path=True, routing_data_folder="data/routing_data", area_text_file="data/Evening_Van_polygon.txt",
                 use_saved_map=False, save_map_structure=True):
        if not os.path.isdir(routing_data_folder):
            os.mkdir(routing_data_folder)
        self.routing_data_folder = routing_data_folder
        self.use_saved_map = use_saved_map
        self.save_map_structure = save_map_structure

        if use_saved_map:
            graphml_filepath = os.path.join(self.routing_data_folder, "graph_structure.graphml")
            self.G = ox.io.load_graphml(graphml_filepath)
        else:
            polygon_coordinates_line = self._load_text_file(file_path=area_text_file)
            latitude_list, longitude_list = self._preprocess_polygon_coordinates(polygon_coordinates_line=polygon_coordinates_line)
            polygon_geometry = Polygon(zip(longitude_list, latitude_list))
            G = self._get_graph_from_polygon(polygon_geometry)
            self.G = ox.truncate.largest_component(G, strongly=True)

        
        if save_map_structure and not use_saved_map:
            graphml_filepath = os.path.join(self.routing_data_folder, "graph_structure.graphml")
            ox.io.save_graphml(self.G, graphml_filepath)

        self.index_dict = self._generate_index_dict()
        self._initialize_map_distances()
        self.colors_w = self._generate_default_node_colors()
        self.map_latitudes, self.map_longitudes = self._extract_map_coordinates()
        self.min_latitude = min(self.map_latitudes)
        self.max_latitude = max(self.map_latitudes)
        self.min_longitude = min(self.map_longitudes)
        self.max_longitude = max(self.map_longitudes)

        self.shortest_paths, self.shortest_paths_time = self._initialize_shortest_path(initialize_shortest_path=initialize_shortest_path)
    
    def _load_text_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                contents = file.readline()
        except FileNotFoundError:
            contents = None
            print("File not found.")
        except Exception as e:
            contents = None
            print(f"An error occurred: {e}")
        
        return contents
    
    def _preprocess_polygon_coordinates(self, polygon_coordinates_line):
        latitude_list = []
        longitude_list = []
        polygon_elements = polygon_coordinates_line.split(" ")
        polygon_elements = polygon_elements[2:-1]

        for polygon_element in polygon_elements:
            coordinates = polygon_element.split(",")
            latitude_list.append(float(coordinates[1]))
            longitude_list.append(float(coordinates[0]))
        
        return latitude_list, longitude_list
    
    def _get_graph_from_polygon(self, polygon):
        G = ox.graph.graph_from_polygon(polygon=polygon,  network_type="drive")
        return G

    def _generate_index_dict(self):
        index_dict = dict()
        for i, key in enumerate(self.G.nodes.keys()):
            index_dict[key] = i
        return index_dict
    
    def _initialize_map_distances(self):
        hwy_speeds = {"residential": 25, "secondary": 40, "tertiary": 60}
        self.G = ox.add_edge_speeds(self.G, hwy_speeds)
        self.G = ox.add_edge_travel_times(self.G)

        edges = ox.convert.graph_to_gdfs(self.G, nodes=False)
        edges["rounded_travel_time"] = edges.apply(lambda row: max(round(row["travel_time"]), 1), axis=1)
        nx.set_edge_attributes(self.G, values=edges["rounded_travel_time"], name="rounded_travel_time")
    
    def _initialize_shortest_path(self, weight="rounded_travel_time", initialize_shortest_path=False):

        if initialize_shortest_path:
            path_dict = {}
            path_time_dict = {}
            origins = []
            destinations = []
            for origin in self.G.nodes.keys():
                for destination in self.G.nodes.keys():
                    origins.append(origin)
                    destinations.append(destination)
            
            new_origins = origins
            new_destinations = destinations
            shortest_routes = ox.routing.shortest_path(self.G, new_origins, new_destinations, weight=weight, cpus=None)
            for i, shortest_route in enumerate(shortest_routes):
                origin = new_origins[i]
                destination = new_destinations[i]
                path_dict[origin, destination] = shortest_route
                if origin == destination:
                    path_time_dict[origin, destination] = 0
                else:
                    new_route_travel_time = int(sum(ox.routing.route_to_gdf(self.G, shortest_route, weight=weight)[weight]))

                    path_time_dict[origin, destination] = new_route_travel_time
                
                print(str(i) + "/" + str(len(shortest_routes)))
            
            self._save_pkl_object(pkl_name="path_dict.pkl", original_struct=path_dict)
            self._save_pkl_object(pkl_name="path_time_dict.pkl", original_struct=path_time_dict)
            
        else:
            path_dict = self._load_pkl_object(pkl_name="path_dict.pkl")
            path_time_dict = self._load_pkl_object(pkl_name="path_time_dict.pkl")
        
        return path_dict, path_time_dict
    
    def _save_pkl_object(self, pkl_name, original_struct):
        pkl_object_path = os.path.join(self.routing_data_folder, pkl_name)
        with open(pkl_object_path, "wb") as handle:
            pickle.dump(original_struct, handle)

    def _load_pkl_object(self, pkl_name):
        pkl_object_path = os.path.join(self.routing_data_folder, pkl_name)
        with open(pkl_object_path, "rb") as handle:
            original_struct = pickle.load(handle)

        return original_struct
    
    def _generate_default_node_colors(self):
        colors_w = ["w" for i in range(len(self.G.nodes.keys()))]
        return colors_w
    
    def _extract_map_coordinates(self):
        map_latitudes = []
        map_longitudes = []
                
        for node in self.G.nodes:
            map_latitudes.append(self.G.nodes[node]['y'])
            map_longitudes.append(self.G.nodes[node]['x'])
        
        return map_latitudes, map_longitudes
    
    def obtain_shortest_paths_time(self, current_location, next_location):
        return self.shortest_paths_time[current_location, next_location]
                
    
    def save_node_data(self):
        latitudes = []
        longitudes = []
        ids = []
        degrees = []
        for node in self.G.nodes:
            ids.append(node)
            latitudes.append(self.G.nodes[node]['y'])
            longitudes.append(self.G.nodes[node]['x'])
            degrees.append(self.G.nodes[node]['street_count'])

        node_data = dict()
        node_data['id'] = ids
        node_data['latitude'] = latitudes
        node_data['longitude'] = longitudes
        node_data['degree'] = degrees
        df = pd.DataFrame(node_data)
        df.to_csv('data/node_data.csv', index=False)
    
    def save_edge_data(self):
        start_points = []
        end_points = []
        costs = []
        for edge in self.G.edges:
            start_points.append(self.index_dict[edge[0]])
            end_points.append(self.index_dict[edge[1]])
            costs.append(self.G.edges[edge]['rounded_travel_time'])

        edge_data = dict()
        edge_data['start'] = start_points
        edge_data['end'] = end_points
        edge_data['cost'] = costs
        df_edges = pd.DataFrame(edge_data)
        df_edges.to_csv('data/edge_data.csv', index=False)
    
    def plot_standard_map(self, map_file="data/normal_map.png"):
        print(len(self.G.nodes))
        print(len(self.G.edges))
        ox.plot_graph(self.G, node_color='#111111', bgcolor='w', edge_color='#111111', filepath=map_file, save=True, 
                      edge_linewidth=0.5, node_size=3, show=False, close=True)



if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        map_graph = Map_graph()
        map_graph.plot_standard_map()

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))