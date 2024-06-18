from collections import Counter
import os
import functools
import osmnx as ox
from frozendict import frozendict
from Map_graph import Map_graph

class Config_flags:

    def __init__(self, consider_route_time: bool, include_scaling: bool, verbose: bool,
                 process_requests: bool, plot_initial_routes: bool, plot_final_routes: bool,
                 create_vid: bool) -> None:
        self.consider_route_time = consider_route_time
        self.include_scaling = include_scaling
        self.verbose = verbose
        self.process_requests = process_requests
        self.plot_initial_routes = plot_initial_routes
        self.plot_final_routes = plot_final_routes
        self.create_vid = create_vid
    
    def __hash__(self) -> int:
        return hash(tuple((self.consider_route_time, self.include_scaling, self.verbose, \
                          self.process_requests, self.plot_initial_routes, self.plot_final_routes, \
                          self.create_vid)))

class Data_folders:

    def __init__(self, processed_requests_folder_path: str = "data/processed_requests/", 
                 predicted_requests_folder_path: str = "data/predicted_requests/",
                 request_folder_path: str = "data/requests/",
                 routing_data_folder: str = "data/routing_data",
                 area_text_file: str = "data/Evening_Van_polygon.txt",
                 static_results_folder: str = "results/static_routes",
                 dynamic_results_folder: str = "results/dynamic_routes",
                 weather_folder: str = "data/weather",
                 model_data_folder: str = "data/model_data") -> None:
        self._check_and_create_folder(processed_requests_folder_path)
        self.processed_requests_folder_path = processed_requests_folder_path

        self._check_and_create_folder(predicted_requests_folder_path)
        self.predicted_requests_folder_path = predicted_requests_folder_path

        self._check_and_create_folder(routing_data_folder)
        self.routing_data_folder = routing_data_folder

        self.area_text_file = area_text_file
        self.request_folder_path = request_folder_path

        self._check_and_create_folder(static_results_folder)
        self.static_results_folder = static_results_folder
        self._check_and_create_folder(dynamic_results_folder)
        self.dynamic_results_folder = dynamic_results_folder

        self.dynamic_policy_results_folder = dynamic_results_folder

        self.weather_folder = weather_folder

        self._check_and_create_folder(model_data_folder)
        self.model_data_folder = model_data_folder
    
    def _check_and_create_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
    
    def __repr__(self) -> str:
        folder_str = ""
        return folder_str

class Simulator_config:

    def __init__(self, map_object: Map_graph, num_buses: int, bus_capacity: int) -> None:
        # Alston
        depot_latitude = 42.3614251
        depot_longitude = -71.1283633

        # # Cambridge
        # depot_latitude = 42.3818293
        # depot_longitude = -71.1292293

        self.num_buses = num_buses
        self.bus_capacities = [bus_capacity] * num_buses
        depot_longitudes = [depot_longitude] * num_buses
        depot_latitudes = [depot_latitude] * num_buses
        self.initial_bus_locations = ox.nearest_nodes(map_object.G, depot_longitudes, depot_latitudes)
    
    def __repr__(self) -> str:
        operational_range_str = ""
        return operational_range_str

class Date_operational_range:
    def __init__(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> None:
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.start_hour: int = start_hour
        self.end_hour: int = end_hour

    def __repr__(self) -> str:
        operational_range_str = ""
        return operational_range_str

    def __hash__(self) -> int:
        return hash(tuple((self.year, self.month, self.day, self.start_hour, self.end_hour)))
    
class Requests_info:
    def __init__(self, requests_df, start_hour: int) -> None:
        self.requests_pickup_times = {}
        self.request_capacities = {}
        for index, row in requests_df.iterrows():
            self.requests_pickup_times[index] = ((((row["Requested Pickup Time"].hour - start_hour) * 60) + row["Requested Pickup Time"].minute) * 60) + row["Requested Pickup Time"].second
            self.request_capacities[index] = row["Number of Passengers"]

    def __repr__(self) -> str:
        pass

    def __hash__(self) -> int:
        return hash(tuple((frozendict(self.requests_pickup_times), frozendict(self.request_capacities))))

class Dataframe_row:
    def __init__(self, data) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(tuple(self.data))

class Bus_stop_request_pairings:
    def __init__(self, stops_request_pairing: list[dict[str, list[int]]]) -> None:
        self.data: list[dict[str, list[int]]] = stops_request_pairing

    def __repr__(self) -> str:
        out = ''
        for stop in self.data:
            pick = '{p' + str([i for i in stop["pickup"]]) + ' '
            drop = 'd' + str([i for i in stop["dropoff"]]) + '}, '
            out += pick+drop
        return out 

    def __hash__(self) -> int:
        content_list = []
        for entry in self.data:
            dict_tuple = tuple(sorted((k, tuple(v)) for k, v in entry.items()))
            content_list.append(dict_tuple)
        return hash(tuple(content_list))
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def pop(self, index: int):
        return self.data.pop(index)
    

class Routing_plan:

    def __init__(self, bus_stops: list[int], stops_wait_times: list[int], 
                 stops_request_pairing: Bus_stop_request_pairings, 
                 assignment_cost: int, start_time: int, route: list[int], 
                 route_edge_times: list[int], route_stop_wait_time: list[int]) -> None:
        self.bus_stops = bus_stops
        self.stops_wait_times = stops_wait_times
        self.stops_request_pairing = stops_request_pairing
        self.assignment_cost = assignment_cost
        self.start_time = start_time
        self.route = route
        self.route_edge_time = route_edge_times
        self.route_stop_wait_time = route_stop_wait_time
    
    def update_routes(self, route: list[int], route_edge_times: list[int], route_stop_wait_time: list[int]):
        self.route = route
        self.route_edge_time = route_edge_times
        self.route_stop_wait_time = route_stop_wait_time

    def __hash__(self) -> int:
        return hash(tuple((tuple(self.bus_stops), tuple(self.stops_wait_times), self.stops_request_pairing, \
                           self.assignment_cost, self.start_time, tuple(self.route), tuple(self.route_edge_time), \
                            tuple(self.route_stop_wait_time))))
    
    def get_unpicked_req_ids(self, stop_index: int) -> list[int]:
        '''SLICED, inplace, generator to iterate over all requests that have not been
        picked up so far in the route'''
        if isinstance(self.stops_request_pairing, list):
            print('wrong')
        combine = [stop['pickup'] + stop['dropoff']
                   for stop in self.stops_request_pairing.data[stop_index:]]
        flatten = [item for sublist in combine for item in sublist if item != -1]
        counts = Counter(flatten)
        repeated_values = [request for request,
                           count in counts.items() if count == 2]
        return repeated_values

    
    def __lt__(self, other: 'Routing_plan') -> bool:
        '''compare two Route instances'''
        return self.assignment_cost < other.assignment_cost
    
    def __repr__(self) -> str:
        repr_str = "(routing_plan=" + str(self.stops_request_pairing) + ')'
        # repr_str = "bus_stops=" + str(self.bus_stops) + ', '
        # repr_str += "Cost=" + str(self.assignment_cost) + ', '
        # repr_str += "Wait Times=" + str(self.stops_wait_times) + ')'
        return repr_str        
   
class Bus_fleet:

    def __init__(self, routing_plans: list[Routing_plan]) -> None:
        self.routing_plans = routing_plans

    def __hash__(self) -> int:
        return hash(tuple(self.routing_plans))
    
class Completed_requests_stats:

    def __init__(self, combined_requests_df, request_status_label = "Request Status", completed_field = "Completed",
                 ride_duration_label = "Ride Duration", number_of_passengers_label = "Number of Passengers", 
                 passenger_wait_time_label = "On-demand ETA"):
        completed_requests_df = combined_requests_df[combined_requests_df[request_status_label] == completed_field]
        number_of_requests = completed_requests_df[number_of_passengers_label].count()
        number_of_serviced_passengers = completed_requests_df[number_of_passengers_label].sum()

        avg_wait_time_at_station = completed_requests_df[passenger_wait_time_label].sum()/completed_requests_df[passenger_wait_time_label].count()
        avg_wait_time_at_station = avg_wait_time_at_station * 60

        avg_time_in_bus = completed_requests_df[ride_duration_label].sum()/completed_requests_df[ride_duration_label].count()
        avg_time_in_bus = avg_time_in_bus * 60

        self.number_of_serviced_requests = number_of_requests
        self.number_of_serviced_passengers = number_of_serviced_passengers
        self.passenger_wait_times = (completed_requests_df[passenger_wait_time_label]* 60).tolist()
        self.avg_wait_time_at_station = avg_wait_time_at_station
        self.avg_time_in_bus = avg_time_in_bus
    
    def __repr__(self) -> str:
        repr_str = "Number of requests serviced = " + str(self.number_of_serviced_requests) + "\n"
        repr_str += "Number of passengers serviced = " + str(self.number_of_serviced_passengers) + "\n"
        repr_str += "Average Passenger Wait Time at the origin station = " + str(self.avg_wait_time_at_station) + "\n"
        repr_str += "Average Passenger Trip Time = " + str(self.avg_time_in_bus) + "\n"
        repr_str += "Passenger Wait Times at the station = " + str(self.passenger_wait_times) + "\n"
        repr_str + "\n"
        
        return repr_str
    
class Failed_requests_stats:

    def __init__(self, combined_requests_df, request_status_label = "Request Status", unaccepted_proposal_field = "Unaccepted Proposal",
                 number_of_passengers_label = "Number of Passengers", passenger_wait_time_label = "On-demand ETA",
                 cancel_field = "Cancel") -> None:
        
        unaccepted_requests_df = combined_requests_df[combined_requests_df[request_status_label] == unaccepted_proposal_field]
        number_of_unaccepted_requests = unaccepted_requests_df[number_of_passengers_label].count()
        number_of_unaccepted_passengers = unaccepted_requests_df[number_of_passengers_label].sum()
        avg_wait_time_for_unaccepted_requests = unaccepted_requests_df[passenger_wait_time_label].sum()/unaccepted_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_unaccepted_requests = avg_wait_time_for_unaccepted_requests * 60

        fleet_canceled_requests_df = combined_requests_df[(combined_requests_df[request_status_label]) == cancel_field]
        number_of_fleet_cancel_requests = fleet_canceled_requests_df[number_of_passengers_label].count()
        number_of_fleet_cancel_passengers = fleet_canceled_requests_df[number_of_passengers_label].sum()
        avg_wait_time_for_fleet_canceled_requests = fleet_canceled_requests_df[passenger_wait_time_label].sum()/fleet_canceled_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_fleet_canceled_requests = avg_wait_time_for_fleet_canceled_requests * 60

        canceled_requests_df = combined_requests_df[(combined_requests_df[request_status_label]).isin([unaccepted_proposal_field, cancel_field])]
        number_of_canceled_requests = canceled_requests_df[number_of_passengers_label].count()
        number_of_canceled_passengers = canceled_requests_df[number_of_passengers_label].sum()
        avg_wait_time_for_canceled_requests = canceled_requests_df[passenger_wait_time_label].sum()/canceled_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_canceled_requests = avg_wait_time_for_canceled_requests * 60
        wait_times_for_canceled_requests = (canceled_requests_df[passenger_wait_time_label] * 60).to_list()

        self.number_of_unaccepted_requests = number_of_unaccepted_requests
        self.number_of_unaccepted_passengers = number_of_unaccepted_passengers
        self.avg_wait_time_for_unaccepted_requests = avg_wait_time_for_unaccepted_requests

        self.number_of_fleet_cancel_requests = number_of_fleet_cancel_requests
        self.number_of_fleet_cancel_passengers = number_of_fleet_cancel_passengers
        self.avg_wait_time_for_canceled_requests = avg_wait_time_for_fleet_canceled_requests

        self.total_number_of_failed_requests = number_of_canceled_requests
        self.total_number_of_failed_passengers = number_of_canceled_passengers
        self.avg_wait_time_for_failed_requests = avg_wait_time_for_canceled_requests
        self.wait_times_for_canceled_requests = wait_times_for_canceled_requests
    
    def __repr__(self) -> str:
        repr_str = "Number of unaccepted  requests = " + str(self.number_of_unaccepted_requests) + "\n"
        repr_str += "Number of unaccepted passengers = " + str(self.number_of_unaccepted_passengers) + "\n"
        repr_str += "Average Wait Time for unaccepted requests = " + str(self.avg_wait_time_for_unaccepted_requests) + "\n"
        repr_str + "\n"
        repr_str +="Number of fleet canceled requests = " + str(self.number_of_fleet_cancel_requests) + "\n"
        repr_str += "Number of fleet canceled passengers = " + str(self.number_of_fleet_cancel_passengers) + "\n"
        repr_str += "Average Wait Time for fleet canceled requests = " + str(self.avg_wait_time_for_canceled_requests) + "\n"
        repr_str + "\n"
        repr_str +="Total number of canceled requests = " + str(self.total_number_of_failed_requests) + "\n"
        repr_str += "Total number of canceled passengers = " + str(self.total_number_of_failed_passengers) + "\n"
        repr_str += "Average Wait Time for canceled requests = " + str(self.avg_wait_time_for_failed_requests) + "\n"
        repr_str += "Wait Times at the station for canceled requests = " + str(self.wait_times_for_canceled_requests) + "\n"
        repr_str + "\n"

        return repr_str

