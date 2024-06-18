"""Policies.py

This module has the classes that define all routing algorithms proposed in the paper.

"""
import os
import copy
import pandas as pd
from scipy.optimize import linear_sum_assignment

from Plot_utils import Plot_utils
from Insertion_procedure import Request_Insertion_Procedure
from Requests_predictor import Request_Prediction_Handler

from State import State
from Map_graph import Map_graph

from Data_structures import Config_flags, Data_folders, Simulator_config, Date_operational_range, Bus_fleet 
from Data_structures import Routing_plan, Bus_stop_request_pairings, Requests_info, Dataframe_row

class Static_Route_Creation_Heuristic:
    
    def __init__(self, map_graph: Map_graph, data_folders: Data_folders, simulator_config: Simulator_config, perfect_accuracy: bool = True):
        
        self.map_graph = map_graph
        self.initial_bus_locations = copy.deepcopy(simulator_config.initial_bus_locations)
        self.bus_capacities = simulator_config.bus_capacities
        self.num_buses = simulator_config.num_buses
        new_results_folder = os.path.join(data_folders.static_results_folder, "heuristic")
        if not os.path.isdir(new_results_folder):
            os.mkdir(new_results_folder)
        self.results_folder = new_results_folder
        self.total_cost = 0
        self.req_pred_handler = Request_Prediction_Handler(data_folders=data_folders,
                                                           perfect_accuracy=perfect_accuracy)
        self.req_insert = Request_Insertion_Procedure(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=simulator_config.num_buses)
        self.bus_stops = []
        self.prev_bus_stops = []
        self.routes = []
        self.routes_edge_time = []
        self.routes_stop_wait_time = []
        self.stops_wait_times = []
        self.prev_stops_wait_times = []
        self.request_assignment = []
        self.current_assignment = []
        self.assignment_cost = []
        self.prev_assignment_cost = []
        self.total_assignment_cost = []
        self.stops_request_pairing = []
        self.prev_stops_request_pairing = []
        self.start_time = []
        self.prev_start_time = []
        self._initialize_buses()

    def _initialize_buses(self):
        for bus_index in range(self.num_buses):
            initial_stops = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_route = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_wait_times = [0, 0]
            initial_stops_request_pairing = [{"pickup": [-1], "dropoff": [-1]}, {"pickup": [-1], "dropoff": [-1]}]
            assignment_cost = 0

            self.bus_stops.append(initial_stops)
            self.prev_bus_stops.append(initial_stops)

            self.routes.append(initial_route)
            self.routes_edge_time.append(0)
            self.routes_stop_wait_time.append(initial_wait_times)

            self.stops_wait_times.append(initial_wait_times)
            self.prev_stops_wait_times.append(initial_wait_times)

            self.request_assignment.append({})

            self.current_assignment.append([])

            self.assignment_cost.append(assignment_cost)
            self.prev_assignment_cost.append(assignment_cost)
            self.total_assignment_cost.append(assignment_cost)

            self.stops_request_pairing.append(initial_stops_request_pairing)
            self.prev_stops_request_pairing.append(initial_stops_request_pairing)

            self.start_time.append(0)
            self.prev_start_time.append(0)

    def retrieve_routes(self):
        return self.routes
    
    def retrieve_bus_stops(self):
        return self.bus_stops
    
    def retrieve_stops_wait_times(self):
        return self.stops_wait_times
    
    def retrieve_request_assignment(self):
        return self.request_assignment
    
    def retrieve_stops_request_pairing(self):
        return self.stops_request_pairing
    
    def retrieve_route_info_for_bus(self, bus_index):
        assignment_cost = self.assignment_cost[bus_index]
        bus_stops = self.bus_stops[bus_index]
        stops_wait_time =  self.stops_wait_times[bus_index]
        request_assignment = self.request_assignment[bus_index]
        stops_request_pair = self.stops_request_pairing[bus_index]
        routes = self.routes[bus_index]
        routes_edge_times = self.routes_edge_time[bus_index]
        routes_stop_wait_time = self.routes_stop_wait_time[bus_index]
        return assignment_cost, bus_stops, stops_wait_time, request_assignment, stops_request_pair, routes, routes_edge_times, routes_stop_wait_time
    
    def retrieve_all_info(self):
        assignment_cost = self.assignment_cost
        bus_stops = self.bus_stops
        stops_wait_time =  self.stops_wait_times
        request_assignment = self.request_assignment
        stops_request_pair = self.stops_request_pairing
        routes = self.routes
        routes_edge_times = self.routes_edge_time
        routes_stop_wait_time = self.routes_stop_wait_time
        bus_start_times = self.start_time
        return assignment_cost, bus_stops, stops_wait_time, request_assignment, stops_request_pair, routes, routes_edge_times, routes_stop_wait_time, bus_start_times
    
    def _get_static_insertion_cost_for_single_request(self, current_start_time, bus_index, request_index, requests_pickup_times, request_row, stops_sequence, 
                                                      stops_wait_time, stop_request_pairing, request_capacities, consider_route_time=True, approximate=False, include_scaling=False):
        request_origin = request_row["Origin Node"]
        request_destination = request_row["Destination Node"]
            
        insertion_result = self.req_insert.static_insertion(current_start_time=current_start_time,
                                                            bus_capacity=self.bus_capacities[bus_index],
                                                            stops_sequence=stops_sequence, 
                                                            stops_wait_time=stops_wait_time, 
                                                            stop_request_pairing=stop_request_pairing,
                                                            request_capacities=request_capacities, 
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination,
                                                            request_index=request_index,
                                                            requests_pickup_times=requests_pickup_times,
                                                            consider_route_time=consider_route_time,
                                                            approximate=approximate,
                                                            include_scaling=include_scaling)
            
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = insertion_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time
    
    def _get_static_insertion_cost(self, requests_df, requests_pickup_times, request_capacities, bus_index, cost_row, stop_row, wait_time_row, 
                                   stop_req_pair_row, req_index_row, start_time_row, consider_route_time=False, approximate=False, include_scaling=False):
        stops_sequence = self.bus_stops[bus_index]
        stops_wait_time = self.stops_wait_times[bus_index]
        stop_request_pairing = self.stops_request_pairing[bus_index]
        current_start_time = self.start_time[bus_index]

        for index, row, in requests_df.iterrows():
            insertion_result = self._get_static_insertion_cost_for_single_request(current_start_time=current_start_time,
                                                                                  bus_index=bus_index, 
                                                                                    request_index=index, 
                                                                                    requests_pickup_times=requests_pickup_times,
                                                                                    request_row=row, 
                                                                                    stops_sequence=stops_sequence,
                                                                                    stops_wait_time=stops_wait_time, 
                                                                                    stop_request_pairing=stop_request_pairing,
                                                                                    request_capacities=request_capacities,
                                                                                    consider_route_time=consider_route_time,
                                                                                    approximate=approximate,
                                                                                    include_scaling=include_scaling)
            total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = insertion_result
            cost_row.append(total_dev_cost)
            stop_row.append(full_stop_sequence)
            wait_time_row.append(full_stops_wait_time)
            stop_req_pair_row.append(full_stop_req_pair)
            req_index_row.append(index)
            start_time_row.append(min_start_time)

    def _generate_cost_for_1D_assignment(self, requests_df, requests_pickup_times, request_capacities, 
                                         consider_route_time=False, approximate=False, include_scaling=False):

        cost_matrix = []
        stop_matrix = []
        wait_time_matrix = []
        stop_req_pair_matrix = []
        req_index_matrix = []
        bus_index_list = []
        start_time_matrix = []

        for bus_index in range(self.num_buses):
            cost_row = []
            stop_row = []
            wait_time_row = []
            stop_req_pair_row = []
            req_index_row = []
            start_time_row = []
            
            self._get_static_insertion_cost(requests_df=requests_df, 
                                            requests_pickup_times=requests_pickup_times,
                                            request_capacities=request_capacities,
                                            bus_index=bus_index, 
                                            cost_row=cost_row,
                                            stop_row=stop_row,
                                            wait_time_row=wait_time_row,
                                            stop_req_pair_row=stop_req_pair_row,
                                            req_index_row=req_index_row, 
                                            start_time_row=start_time_row,
                                            consider_route_time=consider_route_time,
                                            approximate=approximate,
                                            include_scaling=include_scaling)
            
            if len(cost_row) > 0:
                bus_index_list.append(bus_index)
                cost_matrix.append(cost_row)
                stop_matrix.append(stop_row)
                wait_time_matrix.append(wait_time_row)
                stop_req_pair_matrix.append(stop_req_pair_row)
                req_index_matrix.append(req_index_row)
                start_time_matrix.append(start_time_row)
            
        
        return cost_matrix, stop_matrix, wait_time_matrix, stop_req_pair_matrix, req_index_matrix, bus_index_list, start_time_matrix
    
    def _populate_assignment_tracking_based_on_1D_assignment(self, row_indices, col_indices, cost_matrix, stop_matrix, req_index_matrix, 
                                                             wait_time_matrix, stop_req_pair_matrix, bus_index_list, start_time_matrix, 
                                                             combined_requests_df):
        assignment_dict = {}
        serviced_bus_indices = set()
        for assignment_index, task_assignment in zip(row_indices, col_indices):
            bus_index = bus_index_list[assignment_index]
            serviced_bus_indices.add(bus_index)
            assignment_dict[bus_index] = cost_matrix[assignment_index][task_assignment]
            req_index = req_index_matrix[assignment_index][task_assignment]
            self.current_assignment[bus_index] = [req_index]
            self.assignment_cost[bus_index] = cost_matrix[assignment_index][task_assignment]
            self.bus_stops[bus_index] = stop_matrix[assignment_index][task_assignment]
            self.stops_wait_times[bus_index] = wait_time_matrix[assignment_index][task_assignment]
            self.stops_request_pairing[bus_index] = stop_req_pair_matrix[assignment_index][task_assignment]
            self.request_assignment[bus_index][req_index] = combined_requests_df.loc[req_index]
            combined_requests_df = combined_requests_df.drop(req_index)
            self.start_time[bus_index] = start_time_matrix[assignment_index][task_assignment]
        
        for bus_index in range(self.num_buses):
            if bus_index in serviced_bus_indices:
                continue
            else:
                self.current_assignment[bus_index] = []
                self.assignment_cost[bus_index] = 0

        
        sorted_assignment_dict = dict(sorted(assignment_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_assignment_dict, combined_requests_df
    
    def _determine_possible_reassignment(self, requests_pickup_times, request_capacities, bus_index, current_request_index_list, 
                                         consider_route_time=True, approximate=False, include_scaling=False, verbose=False):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_bus_stops = []
        min_stops_wait = []
        min_stops_request_pairing = []
        for secondary_bus_index in range(self.num_buses):
            if secondary_bus_index == bus_index:
                stops_sequence = self.bus_stops[secondary_bus_index]
                stops_wait_time = self.stops_wait_times[secondary_bus_index]
                stop_request_pairing = self.stops_request_pairing[secondary_bus_index]
                total_dev_cost = self.assignment_cost[secondary_bus_index]
                current_start_time = self.start_time[secondary_bus_index]

                if verbose:
                    print("Stop Request Pairing = " + str(stop_request_pairing))
                    print("Stop wait time = " + str(stops_wait_time))
            else:
                total_dev_cost = 0
                stops_sequence = self.bus_stops[secondary_bus_index]
                stops_wait_time = self.stops_wait_times[secondary_bus_index]
                stop_request_pairing = self.stops_request_pairing[secondary_bus_index]
                current_start_time = self.start_time[secondary_bus_index]

                if verbose:
                    print("Stop Request Pairing = " + str(stop_request_pairing))
                    print("Stop wait time = " + str(stops_wait_time))

                for request_index in current_request_index_list:
                    request_row = self.request_assignment[bus_index][request_index]

                    insertion_result = self._get_static_insertion_cost_for_single_request(current_start_time=current_start_time,
                                                                                          bus_index=secondary_bus_index, 
                                                                                            request_index=request_index,
                                                                                            requests_pickup_times=requests_pickup_times,
                                                                                            request_row=request_row,
                                                                                            stops_sequence=stops_sequence,
                                                                                            stops_wait_time=stops_wait_time,
                                                                                            stop_request_pairing=stop_request_pairing,
                                                                                            request_capacities=request_capacities,
                                                                                            consider_route_time=consider_route_time,
                                                                                            approximate=approximate,
                                                                                            include_scaling=include_scaling)
                        
                    
                    dev_cost, stops_sequence, stops_wait_time, stop_request_pairing, current_start_time = insertion_result

                    if verbose:
                        print("Stop Request Pairing = " + str(stop_request_pairing))
                        print("Stop wait time = " + str(stops_wait_time))

                    total_dev_cost += dev_cost

            if verbose:
                print("Cost of moving request from bus " + str(bus_index) + " to " + str(secondary_bus_index) +": " + str(total_dev_cost))

            if total_dev_cost < min_assignment_cost:
                min_assignment_cost = total_dev_cost
                min_bus_index = secondary_bus_index
                min_bus_stops = stops_sequence
                min_stops_wait = stops_wait_time
                min_stops_request_pairing = stop_request_pairing
                min_start_time = current_start_time
        
        return min_bus_index, min_assignment_cost, min_bus_stops, min_stops_wait, min_stops_request_pairing, min_start_time
    
    def _reverse_assignment_to_prev_assignment(self, bus_index):
        self.assignment_cost[bus_index] = 0
        self.start_time[bus_index] = self.prev_start_time[bus_index]
        self.bus_stops[bus_index] = self.prev_bus_stops[bus_index]
        self.stops_wait_times[bus_index] = self.prev_stops_wait_times[bus_index]
        self.stops_request_pairing[bus_index] = self.prev_stops_request_pairing[bus_index]

    def _update_current_assignment_with_reassignment(self, min_bus_index, min_assignment_cost, min_bus_stops, min_stops_wait, 
                                                     min_stops_request_pairing, min_start_time):
        self.assignment_cost[min_bus_index] += min_assignment_cost
        self.bus_stops[min_bus_index] = min_bus_stops
        self.stops_wait_times[min_bus_index] = min_stops_wait
        self.stops_request_pairing[min_bus_index] = min_stops_request_pairing
        self.start_time[min_bus_index] = min_start_time

    
    def _reset_assignment_tracking_for_next_iteration(self):
        for bus_index in range(self.num_buses):
            self.assignment_cost[bus_index] = 0
            self.current_assignment[bus_index] = []
            self.prev_start_time[bus_index] = self.start_time[bus_index]
            self.prev_bus_stops[bus_index] = self.bus_stops[bus_index]
            self.prev_stops_wait_times[bus_index] = self.stops_wait_times[bus_index]
            self.prev_stops_request_pairing[bus_index] = self.stops_request_pairing[bus_index]
    
    def _generate_routes_from_stops(self):
        for bus_index in range(self.num_buses):
            bus_route = []
            bus_route_edge_time = []
            routes_stop_wait_time= []
            current_bus_stops = self.bus_stops[bus_index]
            current_stops_wait_time = self.stops_wait_times[bus_index]
            for bus_stop_index in range(len(current_bus_stops)-1):
                origin_stop = current_bus_stops[bus_stop_index]
                destination_bus_stop = current_bus_stops[bus_stop_index+1]
                origin_wait_time = current_stops_wait_time[bus_stop_index]

                shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
                shortest_path_wait_time = []
                stops_wait_time = []
                for node_index in range(len(shortest_path)-1):
                    edge_origin = shortest_path[node_index]
                    edge_destination = shortest_path[node_index + 1]
                    edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                    shortest_path_wait_time.append(edge_time)
                    if node_index == 0:
                        stops_wait_time.append(origin_wait_time)
                    else:
                        stops_wait_time.append(0)

                if len(shortest_path_wait_time) == 0:
                    shortest_path_wait_time.append(0)
                    stops_wait_time.append(origin_wait_time)
                    bus_route += shortest_path
                else:
                    bus_route += shortest_path[:-1]
                bus_route_edge_time += shortest_path_wait_time
                routes_stop_wait_time += stops_wait_time
            

            bus_route += [current_bus_stops[-1]]
            routes_stop_wait_time += [current_stops_wait_time[-1]]
            self.routes[bus_index] = bus_route
            self.routes_stop_wait_time[bus_index] = routes_stop_wait_time
            self.routes_edge_time[bus_index] = bus_route_edge_time
    
    def _reassign_requests(self, sorted_assignment_dict, requests_pickup_times, request_capacities, consider_route_time=True, 
                           approximate=False, include_scaling=False, verbose=False):
        if verbose:
            print("Assignment cost = " + str(self.assignment_cost))
        for bus_index in sorted_assignment_dict.keys():
            current_request_index_list = self.current_assignment[bus_index]

            reassignment_result = self._determine_possible_reassignment(requests_pickup_times=requests_pickup_times, 
                                                                        request_capacities=request_capacities,
                                                                        bus_index=bus_index, 
                                                                        current_request_index_list=current_request_index_list, 
                                                                        consider_route_time=consider_route_time,
                                                                        approximate=approximate,
                                                                        include_scaling=include_scaling)
            
            
            min_bus_index, min_assignment_cost, min_bus_stops, min_stops_wait, min_stops_request_pairing, min_start_time = reassignment_result
            
            if min_bus_index != bus_index:
                if verbose:
                    print("Request for bus " + str(bus_index) + " got moved to bus " + str(min_bus_index))
                self._reverse_assignment_to_prev_assignment(bus_index=bus_index)

                self._update_current_assignment_with_reassignment(min_bus_index=min_bus_index, min_assignment_cost=min_assignment_cost,
                                                                    min_bus_stops=min_bus_stops, min_stops_wait=min_stops_wait,
                                                                    min_stops_request_pairing=min_stops_request_pairing,
                                                                    min_start_time=min_start_time)
                for request_index in current_request_index_list:
                    self.request_assignment[min_bus_index][request_index] = self.request_assignment[bus_index][request_index]
                    self.request_assignment[bus_index].pop(request_index)
                
                self.current_assignment[min_bus_index] += current_request_index_list
                self.current_assignment[bus_index] = []

            else:
                if verbose:
                    print("Request for bus " + str(min_bus_index) + " stays in the bus")
                    print("Assignment cost = " + str(self.assignment_cost))
        
        for bus_index in range(self.num_buses):
            self.total_assignment_cost[bus_index] += self.assignment_cost[bus_index]
        
        stage_cost = sum(self.assignment_cost)
        self.total_cost += stage_cost

        if verbose:
            print("Assignment cost = " + str(self.assignment_cost))
        
        self._generate_routes_from_stops()

        self._reset_assignment_tracking_for_next_iteration()
    
    def _extract_requests(self, year_of_interest, month_of_interest, day_of_interest, hour_of_interest, start_minute, end_minute):
        scheduled_requests_df, online_requests_df = self.req_pred_handler.get_requests_for_given_minute_range(year=year_of_interest,
                                                                                                              month=month_of_interest,
                                                                                                              day=day_of_interest,
                                                                                                              hour=hour_of_interest,
                                                                                                              start_minute=start_minute,
                                                                                                              end_minute=end_minute)
        
        combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])

        return combined_requests_df
        
        
    def assign_requests_and_create_routes(self, date_operational_range: Date_operational_range, config_flags: Config_flags):
        if config_flags.plot_initial_routes:
            self.plot_utils.reset_frame_number()
        
        requests_pickup_times = {}
        request_capacities = {}

        # minute_intervals = [(0,19), (20, 39), (40, 59)]
        # minute_intervals = [(0,14), (15,29), (30,44), (45,59)]
        # minute_intervals = [(0,9), (10,19), (20,29), (30,39), (40,49), (50,59)]
        minute_intervals = [(0,4), (5,9), (10,14), (15,19), (20,24), (25,29), (30,34), (35,39), (40,44), (45,49), (50,54), (55,59)]

        hour_range, day_range, month_range, year_range = self.req_pred_handler.generate_operating_ranges(date_operational_range=date_operational_range)
        
        for i, hour_of_interest in enumerate(hour_range):   

            for time_interval in minute_intervals:

                combined_requests_df = self._extract_requests(year_of_interest=year_range[i],
                                                              month_of_interest=month_range[i],
                                                              day_of_interest=day_range[i],
                                                              hour_of_interest=hour_of_interest,
                                                              start_minute=time_interval[0],
                                                              end_minute=time_interval[1])
                requests_to_be_serviced = combined_requests_df

                for index, row in requests_to_be_serviced.iterrows():
                    requests_pickup_times[index] = ((((row["Requested Pickup Time"].hour - date_operational_range.start_hour) * 60) \
                                                     + row["Requested Pickup Time"].minute) * 60) + row["Requested Pickup Time"].second
                    request_capacities[index] = row["Number of Passengers"]

                while not requests_to_be_serviced.empty:
                    assignment_matrices = self._generate_cost_for_1D_assignment(requests_df=requests_to_be_serviced,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                request_capacities=request_capacities,
                                                                                consider_route_time=config_flags.consider_route_time,
                                                                                approximate=False,
                                                                                include_scaling=config_flags.include_scaling)
                        
                    cost_matrix, stop_matrix, wait_time_matrix, stop_req_pair_matrix, req_index_matrix, bus_index_list, start_time_matrix = assignment_matrices
                    if len(bus_index_list) > 0:
                            row_indices, col_indices = linear_sum_assignment(cost_matrix)
                            sorted_assignment_dict, requests_to_be_serviced = self._populate_assignment_tracking_based_on_1D_assignment(row_indices=row_indices,
                                                                                                                                        col_indices=col_indices,
                                                                                                                                        cost_matrix=cost_matrix,
                                                                                                                                        stop_matrix=stop_matrix,
                                                                                                                                        req_index_matrix=req_index_matrix,
                                                                                                                                        wait_time_matrix=wait_time_matrix,
                                                                                                                                        stop_req_pair_matrix=stop_req_pair_matrix,
                                                                                                                                        bus_index_list=bus_index_list,
                                                                                                                                        start_time_matrix=start_time_matrix,
                                                                                                                                        combined_requests_df=requests_to_be_serviced)
                            
                            if config_flags.plot_initial_routes:
                                self.plot_utils.plot_routes_before_assignment_offline(map_object=self.map_graph, current_assignment=self.current_assignment, 
                                                                            request_assignment=self.request_assignment, prev_bus_stops=self.prev_bus_stops,
                                                                            prev_bus_routes=self.routes, bus_locations=self.initial_bus_locations,
                                                                            folder_path=self.results_folder)
                                
                            self._reassign_requests(sorted_assignment_dict=sorted_assignment_dict,
                                                    requests_pickup_times=requests_pickup_times, 
                                                    request_capacities=request_capacities,
                                                    consider_route_time=config_flags.consider_route_time, 
                                                    approximate=False, 
                                                    include_scaling=config_flags.include_scaling)

                            if config_flags.plot_initial_routes:
                                self.plot_utils.plot_routes_after_assignment_offline(map_object=self.map_graph, outstanding_requests={}, current_bus_stops=self.bus_stops,
                                                                            current_bus_routes=self.routes, bus_locations=self.initial_bus_locations,
                                                                            folder_path=self.results_folder)

        return self.total_cost, self.total_assignment_cost, requests_pickup_times, request_capacities

class Static_Route_Creation_Rollout:
    
    def __init__(self, map_graph: Map_graph, data_folders: Data_folders, simulator_config: Simulator_config, perfect_accuracy: bool = True):
        
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.bus_capacities = simulator_config.bus_capacities
        self.num_buses = simulator_config.num_buses
        new_results_folder = os.path.join(data_folders.static_results_folder, "rollout")
        if not os.path.isdir(new_results_folder):
            os.mkdir(new_results_folder)
        self.results_folder = new_results_folder
        self.total_cost = 0
        self.req_pred_handler = Request_Prediction_Handler(data_folders=data_folders,
                                                           perfect_accuracy=perfect_accuracy)
        self.req_insert = Request_Insertion_Procedure_baseline1(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=simulator_config.num_buses)
        self.request_assignment = []
        self.current_assignment = []
        routing_plans = self._initialize_buses()
        self.bus_fleet = Bus_fleet(routing_plans=routing_plans)

    def _initialize_buses(self) -> list[Routing_plan]:
        routing_plans = []
        for bus_index in range(self.num_buses):
            self.request_assignment.append({})
            self.current_assignment.append([])
            initial_stops = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_wait_times = [0, 0]
            initial_stops_request_pairing_list = [{"pickup": [-1], "dropoff": [-1]}, {"pickup": [-1], "dropoff": [-1]}]
            initial_stops_request_pairing = Bus_stop_request_pairings(stops_request_pairing=initial_stops_request_pairing_list)
            initial_assignment_cost = 0
            initial_start_time = 0
            initial_route = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_route_edge_time = [0]
            initial_route_stop_wait_time = [0, 0]

            bus_routing_plan = Routing_plan(bus_stops=initial_stops,
                                            stops_wait_times=initial_wait_times,
                                            stops_request_pairing=initial_stops_request_pairing,
                                            assignment_cost=initial_assignment_cost,
                                            start_time=initial_start_time,
                                            route=initial_route,
                                            route_edge_times=initial_route_edge_time,
                                            route_stop_wait_time=initial_route_stop_wait_time)
            
            routing_plans.append(bus_routing_plan)
        
        return routing_plans
    
    def retrieve_route_info_for_bus(self, bus_index: int) -> Routing_plan:
        return self.bus_fleet.routing_plans[bus_index]
    
    def retrieve_all_info(self):
        return self.bus_fleet, self.request_assignment
    
    def _get_static_insertion_cost_for_single_request(self, bus_index: int, request_index: int, request_row: Dataframe_row, 
                                                      requests_info: Requests_info, local_routing_plan: Routing_plan, 
                                                      config_flags: Config_flags):
        request_origin = request_row.data["Origin Node"]
        request_destination = request_row.data["Destination Node"]
        
        insertion_result = self.req_insert.static_insertion(current_start_time=local_routing_plan.start_time,
                                                            bus_capacity=self.bus_capacities[bus_index],
                                                            stops_sequence=local_routing_plan.bus_stops, 
                                                            stops_wait_time=local_routing_plan.stops_wait_times, 
                                                            stop_request_pairing=local_routing_plan.stops_request_pairing.data,
                                                            request_capacities=requests_info.request_capacities, 
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination,
                                                            request_index=request_index,
                                                            requests_pickup_times=requests_info.requests_pickup_times,
                                                            consider_route_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling)
            
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = insertion_result

        new_assignment_cost = local_routing_plan.assignment_cost + total_dev_cost
        new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)
        new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                        stops_wait_times=full_stops_wait_time,
                                        stops_request_pairing=new_stop_req_pairings,
                                        assignment_cost=new_assignment_cost,
                                        start_time=min_start_time,
                                        route=[],
                                        route_edge_times=[],
                                        route_stop_wait_time=[])

        return total_dev_cost, new_routing_plan
    
    def _generate_routes_from_stops(self):
        for bus_index in range(self.num_buses):
            bus_route = []
            bus_route_edge_time = []
            routes_stop_wait_time= []
            current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
            current_stops_wait_time = self.bus_fleet.routing_plans[bus_index].stops_wait_times
            for bus_stop_index in range(len(current_bus_stops)-1):
                origin_stop = current_bus_stops[bus_stop_index]
                destination_bus_stop = current_bus_stops[bus_stop_index+1]
                origin_wait_time = current_stops_wait_time[bus_stop_index]

                shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
                shortest_path_wait_time = []
                stops_wait_time = []
                for node_index in range(len(shortest_path)-1):
                    edge_origin = shortest_path[node_index]
                    edge_destination = shortest_path[node_index + 1]
                    edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                    shortest_path_wait_time.append(edge_time)
                    if node_index == 0:
                        stops_wait_time.append(origin_wait_time)
                    else:
                        stops_wait_time.append(0)

                if len(shortest_path_wait_time) == 0:
                    shortest_path_wait_time.append(0)
                    stops_wait_time.append(origin_wait_time)
                    bus_route += shortest_path
                else:
                    bus_route += shortest_path[:-1]
                bus_route_edge_time += shortest_path_wait_time
                routes_stop_wait_time += stops_wait_time
            

            bus_route += [current_bus_stops[-1]]
            routes_stop_wait_time += [current_stops_wait_time[-1]]
            self.bus_fleet.routing_plans[bus_index].update_routes(route=bus_route, 
                                                                  route_edge_times=bus_route_edge_time,
                                                                  route_stop_wait_time=routes_stop_wait_time)
    
    def _extract_requests(self, date_operational_range: Date_operational_range):
        scheduled_requests_df, online_requests_df = self.req_pred_handler.\
            get_requests_for_given_date_and_hour_range(date_operational_range=date_operational_range)
        
        combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])

        return combined_requests_df
    
    def _obtain_rollout_cost(self, requests_df, requests_info: Requests_info, bus_fleet: Bus_fleet, config_flags: Config_flags):
        
        local_bus_fleet = copy.deepcopy(bus_fleet)

        for index, row, in requests_df.iterrows():
            min_assignment_cost = float("inf")
            min_bus_index = 0
            min_routing_plan = None
            for bus_index in range(self.num_buses):
                local_routing_plan = local_bus_fleet.routing_plans[bus_index]
                request_row = Dataframe_row(data=row)
                total_dev_cost, new_routing_plan = self._get_static_insertion_cost_for_single_request(bus_index=bus_index, 
                                                                                        request_index=index,
                                                                                        request_row=request_row,
                                                                                        requests_info=requests_info,
                                                                                        local_routing_plan=local_routing_plan,
                                                                                        config_flags=config_flags)

                if total_dev_cost < min_assignment_cost:
                    min_assignment_cost = total_dev_cost
                    min_bus_index = bus_index
                    min_routing_plan = new_routing_plan
            
            local_bus_fleet.routing_plans[min_bus_index] = min_routing_plan
        
        assignment_costs = []
        for routing_plan in local_bus_fleet.routing_plans:
            current_assignment_cost = routing_plan.assignment_cost
            assignment_costs.append(current_assignment_cost)
        
        return assignment_costs
    
    def _insert_single_request(self, bus_index: int, request_index: int, request_row: Dataframe_row, 
                               requests_info: Requests_info, config_flags: Config_flags):
        local_bus_fleet = copy.deepcopy(self.bus_fleet)

        local_routing_plan = local_bus_fleet.routing_plans[bus_index]

        total_dev_cost, new_routing_plan = self._get_static_insertion_cost_for_single_request(bus_index=bus_index, 
                                                                              request_index=request_index,
                                                                              request_row=request_row,
                                                                              requests_info=requests_info,
                                                                              local_routing_plan=local_routing_plan,
                                                                              config_flags=config_flags)
        
        local_bus_fleet.routing_plans[bus_index] = new_routing_plan

        return local_bus_fleet

    
    def _assign_request(self, requests_info: Requests_info, request_index: int, request_row: Dataframe_row, 
                        remaining_requests, config_flags: Config_flags):
        
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        for bus_index in range(self.num_buses):

            bus_fleet = self._insert_single_request(bus_index=bus_index,
                                                    request_index=request_index,
                                                    request_row=request_row,
                                                    requests_info=requests_info,
                                                    config_flags=config_flags)

            rollout_costs = self._obtain_rollout_cost(requests_df=remaining_requests,
                                                      requests_info=requests_info,
                                                      bus_fleet=bus_fleet,
                                                      config_flags=config_flags)
            
            final_assignment_cost = sum(rollout_costs)
            
            if final_assignment_cost < min_assignment_cost:
                min_assignment_cost = final_assignment_cost
                min_bus_index = bus_index
                min_routing_plan = bus_fleet.routing_plans[bus_index]
        
        self.request_assignment[min_bus_index][request_index] = request_row.data
        self.current_assignment[min_bus_index] = [request_index]

        if config_flags.plot_initial_routes:
            prev_bus_stops = []
            prev_bus_routes = []
            for bus_index in range(self.num_buses):
                current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
                current_routes = self.bus_fleet.routing_plans[bus_index].route
                prev_bus_routes.append(current_routes)
                prev_bus_stops.append(current_bus_stops)
            self.plot_utils.plot_routes_before_assignment_offline(map_object=self.map_graph, 
                                                                  current_assignment=self.current_assignment, 
                                                                  request_assignment=self.request_assignment, 
                                                                  prev_bus_stops=prev_bus_stops,
                                                                  prev_bus_routes=prev_bus_routes, 
                                                                  bus_locations=self.initial_bus_locations,
                                                                  folder_path=self.results_folder)
        
        self.bus_fleet.routing_plans[min_bus_index] = min_routing_plan

        self._generate_routes_from_stops()

        if config_flags.plot_initial_routes:
            current_bus_stops_list = []
            current_bus_routes_list = []
            for bus_index in range(self.num_buses):
                current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
                current_routes = self.bus_fleet.routing_plans[bus_index].route
                current_bus_routes_list.append(current_routes)
                current_bus_stops_list.append(current_bus_stops)
            self.plot_utils.plot_routes_after_assignment_offline(map_object=self.map_graph, 
                                                                 outstanding_requests={}, 
                                                                 current_bus_stops=current_bus_stops_list,
                                                                 current_bus_routes=current_bus_routes_list, 
                                                                 bus_locations=self.initial_bus_locations,
                                                                 folder_path=self.results_folder)
    
    def _get_truncated_request_dataframe(self, requests_to_be_serviced, truncation_horizon: int = 50):
        size_of_df = len(requests_to_be_serviced.index)
        if size_of_df == 0:
            return requests_to_be_serviced
        elif size_of_df > truncation_horizon:
            truncation_index = truncation_horizon
        else:
            truncation_index = size_of_df
        truncated_request_df = requests_to_be_serviced.iloc[:truncation_index]

        return truncated_request_df
        
    def assign_requests_and_create_routes(self, date_operational_range: Date_operational_range, config_flags: Config_flags,
                                          truncation_horizon: int = 50):
        if config_flags.plot_initial_routes:
            self.plot_utils.reset_frame_number()

        combined_requests_df = self._extract_requests(date_operational_range=date_operational_range)
        requests_to_be_serviced = copy.deepcopy(combined_requests_df)

        requests_info = Requests_info(requests_df=combined_requests_df, start_hour=date_operational_range.start_hour)

        for index, row in combined_requests_df.iterrows():
            requests_to_be_serviced = requests_to_be_serviced.drop(index)

            truncated_requests_to_be_serviced = self._get_truncated_request_dataframe(requests_to_be_serviced=requests_to_be_serviced,
                                                                                      truncation_horizon=truncation_horizon)
            
            request_row = Dataframe_row(data=row)
            self._assign_request(requests_info=requests_info,
                                 request_index=index,
                                 request_row=request_row,
                                 remaining_requests=truncated_requests_to_be_serviced,
                                 config_flags=config_flags)
        assignment_costs = []

        for routing_plan in self.bus_fleet.routing_plans:
            current_assignment_cost = routing_plan.assignment_cost
            assignment_costs.append(current_assignment_cost)

        self.total_cost = sum(assignment_costs)

        return self.total_cost, assignment_costs, requests_info


class Dynamic_Route_Creation:

    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags):
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure_baseline1(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
    
    def _dropoff_prev_passengers(self, state_object: State, bus_index: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        next_bus_stop_index = state_object.bus_stop_index[bus_index] + 1

        dropoff_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[next_bus_stop_index]["dropoff"]
        for dropoff_request_index in dropoff_request_index_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in new_prev_passengers:
                    del(new_prev_passengers[dropoff_request_index])
                    new_passengers_in_bus -= state_object.request_capacities[dropoff_request_index]
        
        return new_passengers_in_bus, new_prev_passengers
    
    def _pickup_prev_passengers(self, state_object: State, bus_index: int, current_start_time: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        current_bus_stop_index = state_object.bus_stop_index[bus_index]

        pickup_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_bus_stop_index]["pickup"]
        for pickup_request_index in pickup_request_index_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in new_prev_passengers:
                    new_prev_passengers[pickup_request_index] = [state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index], current_start_time]
                    new_passengers_in_bus += state_object.request_capacities[pickup_request_index]
        
        return new_passengers_in_bus, new_prev_passengers
    
    def _get_bus_parameters_of_interest(self, state_object: State, bus_index: int):
        current_step_index = state_object.step_index[bus_index]
        next_bus_location = state_object.bus_fleet.routing_plans[bus_index].route[current_step_index+1]
        current_bus_location = state_object.bus_locations[bus_index]

        current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
        current_edge_time = state_object.bus_fleet.routing_plans[bus_index].route_edge_time[current_step_index]

        current_bus_stop_index = state_object.bus_stop_index[bus_index]
        next_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]
        current_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index]

        if len(state_object.bus_fleet.routing_plans[bus_index].bus_stops) == 2 and \
            state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index] == state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]:
            new_bus_location = current_bus_location
            current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
            current_stop_index = current_bus_stop_index
            passengers_in_bus = state_object.passengers_in_bus[bus_index]
            prev_passengers = state_object.prev_passengers[bus_index]
        else:
            if current_bus_location == current_bus_stop:
                if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    new_bus_location = current_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

                else:
                    new_bus_location = next_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_stop_wait_time
                    new_passengers_in_bus, new_prev_passengers = self._pickup_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                current_start_time=current_start_time,
                                                                                                passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                                prev_passengers=state_object.prev_passengers[bus_index])
                    current_start_time += current_edge_time
                    if next_bus_location == next_bus_stop:
                        current_stop_index = current_bus_stop_index + 1
                        new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                passengers_in_bus=new_passengers_in_bus,
                                                                                                prev_passengers=new_prev_passengers)
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
                    else:
                        current_stop_index = current_bus_stop_index
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
            else:
                new_bus_location = next_bus_location
                current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_edge_time
                if next_bus_location == next_bus_stop:
                    current_stop_index = current_bus_stop_index + 1
                    new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                            prev_passengers=state_object.prev_passengers[bus_index])
                    passengers_in_bus = new_passengers_in_bus
                    prev_passengers = new_prev_passengers
                else:
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

        return current_start_time, current_stop_index, new_bus_location, passengers_in_bus, prev_passengers


    def _get_dynamic_insertion_cost_for_request(self, state_object: State, bus_index: int, request_index: int, request_row, config_flags: Config_flags):
        request_origin = request_row["Origin Node"]
        request_destination = request_row["Destination Node"]

        bus_parameters = self._get_bus_parameters_of_interest(state_object=state_object, bus_index=bus_index)

        current_start_time, current_stop_index, current_location, passengers_in_bus, prev_passengers = bus_parameters

        insertion_result = self.req_insert.dynamic_insertion(current_start_time=current_start_time,
                                                             current_stop_index=current_stop_index,
                                                             bus_location=current_location,
                                                             bus_capacity=state_object.bus_capacities[bus_index],
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stops_sequence=state_object.bus_fleet.routing_plans[bus_index].bus_stops, 
                                                            stops_wait_time=state_object.bus_fleet.routing_plans[bus_index].stops_wait_times, 
                                                            stop_request_pairing=state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data,
                                                            request_capacities=state_object.request_capacities,
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination, 
                                                            requests_pickup_times=state_object.requests_pickup_times,
                                                            request_index=request_index,
                                                            consider_route_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair = insertion_result

        new_assignment_cost = state_object.bus_fleet.routing_plans[bus_index].assignment_cost + total_dev_cost
        new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)

        new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                        stops_wait_times=full_stops_wait_time,
                                        stops_request_pairing=new_stop_req_pairings,
                                        assignment_cost=new_assignment_cost,
                                        start_time=state_object.bus_fleet.routing_plans[bus_index].start_time,
                                        route=[],
                                        route_edge_times=[],
                                        route_stop_wait_time=[])

        
        return total_dev_cost, new_routing_plan

    def _determine_assignment(self, state_object, current_request_index, current_request_row, config_flags: Config_flags):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        for bus_index in range(self.num_buses):
            total_dev_cost, new_routing_plan = self._get_dynamic_insertion_cost_for_request(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            request_index=current_request_index,
                                                                                            request_row=current_request_row,
                                                                                            config_flags=config_flags)

            if total_dev_cost < min_assignment_cost:
                min_assignment_cost = total_dev_cost
                min_bus_index = bus_index
                min_routing_plan = new_routing_plan
        
        return min_bus_index, min_assignment_cost, min_routing_plan
    
    def _generate_route_from_stops(self, current_bus_location, bus_stops, stops_wait_times):
        bus_route = []
        bus_route_edge_time = []
        routes_stop_wait_time= []
        for bus_stop_index in range(len(bus_stops)-1):
            if bus_stop_index == 0:
                origin_stop = current_bus_location
                if bus_stops[bus_stop_index] == current_bus_location:
                    origin_wait_time = stops_wait_times[bus_stop_index]
                else:
                    origin_wait_time = 0
            else:
                origin_stop = bus_stops[bus_stop_index]
                origin_wait_time = stops_wait_times[bus_stop_index]
            destination_bus_stop = bus_stops[bus_stop_index+1]

            shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
            shortest_path_wait_time = []
            stops_wait_time = []
            for node_index in range(len(shortest_path)-1):
                edge_origin = shortest_path[node_index]
                edge_destination = shortest_path[node_index + 1]
                edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                shortest_path_wait_time.append(edge_time)
                if node_index == 0:
                    stops_wait_time.append(origin_wait_time)
                else:
                    stops_wait_time.append(0)

            if len(shortest_path_wait_time) == 0:
                shortest_path_wait_time.append(0)
                stops_wait_time.append(origin_wait_time)
                bus_route += shortest_path
            else:
                bus_route += shortest_path[:-1]
            
            bus_route_edge_time += shortest_path_wait_time
            routes_stop_wait_time += stops_wait_time
        
        bus_route += [bus_stops[-1]]
        routes_stop_wait_time += [stops_wait_time[-1]]
        
        return bus_route, bus_route_edge_time, routes_stop_wait_time

    def assign_requests_and_create_routes(self, state_object: State, requests, config_flags: Config_flags):

        if config_flags.plot_final_routes:
            prev_bus_stops = []
            prev_bus_routes = []
            for bus_index in range(state_object.num_buses):
                current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                prev_bus_routes.append(current_routes)
                prev_bus_stops.append(current_bus_stops)

            self.plot_utils.plot_routes_before_assignment_online(map_object=self.map_graph, 
                                                                 requests=requests, 
                                                                 prev_bus_stops=prev_bus_stops,
                                                                 prev_bus_routes=prev_bus_routes, 
                                                                 bus_locations=state_object.bus_locations,
                                                                 folder_path=self.results_folder)
        
        for request_index, request_row, in requests.iterrows():
            state_object.requests_pickup_times[request_index] = ((((request_row["Requested Pickup Time"].hour - state_object.date_operational_range.start_hour) * 60) \
                                                                  + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            state_object.request_capacities[request_index] = request_row["Number of Passengers"]

        for request_index, request_row, in requests.iterrows():
            min_bus_index, assignment_cost, new_routing_plan = self._determine_assignment(state_object=state_object,
                                                                                    current_request_index=request_index,
                                                                                    current_request_row=request_row, 
                                                                                    config_flags=config_flags)

            current_step_index = state_object.step_index[min_bus_index]
            current_stop_wait_time = state_object.bus_fleet.routing_plans[min_bus_index].route_stop_wait_time[current_step_index]
            if state_object.wait_time_at_the_station[min_bus_index] < current_stop_wait_time:
                current_bus_location=state_object.bus_fleet.routing_plans[min_bus_index].route[state_object.step_index[min_bus_index]]
            else:
                current_bus_location=state_object.bus_fleet.routing_plans[min_bus_index].route[state_object.step_index[min_bus_index]+1]

            route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                bus_stops=new_routing_plan.bus_stops, 
                                                                                                stops_wait_times=new_routing_plan.stops_wait_times)
            
            new_routing_plan.update_routes(route=route,
                                           route_edge_times=bus_route_edge_time,
                                           route_stop_wait_time=routes_stop_wait_time)
            
            state_object.update_state(min_bus_index, 
                                      request_index=request_index,
                                      request_row=request_row,
                                      assignment_cost=assignment_cost,
                                      new_routing_plan=new_routing_plan)
        
        if config_flags.plot_final_routes and not requests.empty:
                current_bus_stops_list = []
                current_bus_routes_list = []
                for bus_index in range(state_object.num_buses):
                    current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                    current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                    current_bus_routes_list.append(current_routes)
                    current_bus_stops_list.append(current_bus_stops)
                self.plot_utils.plot_routes_after_assignment_online(map_object=self.map_graph, 
                                                                    outstanding_requests={}, 
                                                                    current_bus_stops=current_bus_stops_list,
                                                                    current_bus_routes=current_bus_routes_list, 
                                                                    bus_locations=state_object.bus_locations,
                                                                    folder_path=self.results_folder)

class Proactive_Bus_Routing:

    def __init__(self, map_graph: Map_graph, state_object: State, data_folders: Data_folders, simulator_config: Simulator_config, 
                 config_flags: Config_flags, date_operational_range: Date_operational_range, perfect_accuracy: bool = True,
                 truncation_horizon: int = 50):
        
        self.num_buses = simulator_config.num_buses

        self.static_route_creation = Static_Route_Creation_Rollout(map_graph=map_graph,
                                                                   data_folders=data_folders,
                                                                   simulator_config=simulator_config,
                                                                   perfect_accuracy=perfect_accuracy)
        
        self.dynamic_route_creation = Dynamic_Route_Creation(map_graph=map_graph, 
                                                             data_folders=data_folders,
                                                             simulator_config=simulator_config,
                                                             config_flags=config_flags)
        
        self.plot_utils = self.dynamic_route_creation.plot_utils
        
        self.static_route_creation.assign_requests_and_create_routes(date_operational_range=date_operational_range,
                                                                     config_flags=config_flags,
                                                                     truncation_horizon=truncation_horizon)
        
        predicted_bus_fleet, request_assignment = self.static_route_creation.retrieve_all_info()
        
        request_dictionaries = self._generate_request_dictionaries(state_object=state_object)

        requests_pickup_times, request_capacities, combined_pickup_times, combined_request_capacities = request_dictionaries    

        new_assignment_cost = self._obtain_initial_assignment_cost_and_pairing(map_graph=map_graph,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                request_capacities=request_capacities,
                                                                                simulator_config=simulator_config,
                                                                                predicted_bus_fleet=predicted_bus_fleet,
                                                                                request_assignment=request_assignment,
                                                                                config_flags=config_flags)

        state_object.reinitialize_state(initial_assignment_cost=new_assignment_cost,
                                        predicted_bus_fleet=predicted_bus_fleet,
                                        initial_request_capacities=combined_request_capacities,
                                        requests_pickup_times=combined_pickup_times,
                                        simulator_config=simulator_config)

    def _generate_request_dictionaries(self, state_object: State):
        scheduled_requests_df, online_requests_df = self.static_route_creation.req_pred_handler.get_requests_for_given_date_and_hour_range(state_object.date_operational_range)
        requests_pickup_times = {}
        request_capacities = {}
        for request_index, request_row, in scheduled_requests_df.iterrows():
            requests_pickup_times[request_index] = ((((request_row["Requested Pickup Time"].hour - state_object.date_operational_range.start_hour) * 60) \
                                                     + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            request_capacities[request_index] = request_row["Number of Passengers"]
        
        predicted_requests_pickup_times = {}
        predicted_request_capacities = {}
        for request_index, request_row, in online_requests_df.iterrows():
            predicted_requests_pickup_times[-1*request_index] = ((((request_row["Requested Pickup Time"].hour - state_object.date_operational_range.start_hour) * 60) \
                                                                  + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            predicted_request_capacities[-1*request_index] = request_row["Number of Passengers"]

        combined_pickup_times = {**requests_pickup_times, **predicted_requests_pickup_times}
        combined_request_capacities = {**request_capacities, **predicted_request_capacities}

        return requests_pickup_times, request_capacities, combined_pickup_times, combined_request_capacities
        
    def _obtain_initial_assignment_cost_and_pairing(self, map_graph: Map_graph, requests_pickup_times, request_capacities, simulator_config: Simulator_config, 
                                                    predicted_bus_fleet: Bus_fleet, request_assignment, config_flags: Config_flags):
        new_assignment_cost = []
        for bus_index in range(simulator_config.num_buses):
            requests_wait_time = 0
            serviced_requests = {}
            real_time = predicted_bus_fleet.routing_plans[bus_index].start_time
            route_time = 0
            for i in range(len(predicted_bus_fleet.routing_plans[bus_index].bus_stops)-1):
                current_location = predicted_bus_fleet.routing_plans[bus_index].bus_stops[i]
                next_location = predicted_bus_fleet.routing_plans[bus_index].bus_stops[i+1]

                current_request_index_dict = predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]

                for list_index, pickup_request_index in enumerate(pickup_requests_list):
                    if pickup_request_index != -1:
                        if pickup_request_index in requests_pickup_times:
                            request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                            actual_pickup_time = real_time + predicted_bus_fleet.routing_plans[bus_index].stops_wait_times[i]
                            if config_flags.include_scaling:
                                wait_time_at_the_station = (1/simulator_config.bus_capacities[bus_index]) * (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                            else:
                                wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                            requests_wait_time += wait_time_at_the_station
                            serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
                        else:
                            new_pickup_index = -1 * predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["pickup"][list_index]
                            if pickup_request_index in request_assignment[bus_index]:
                                del(request_assignment[bus_index][pickup_request_index])
                            predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["pickup"][list_index] = new_pickup_index
                
                for list_index, dropoff_request_index in enumerate(dropoff_requests_list):
                    if dropoff_request_index != -1:
                        if dropoff_request_index in requests_pickup_times:
                            initial_station = serviced_requests[dropoff_request_index][0]
                            final_station = current_location
                            direct_route_time = map_graph.obtain_shortest_paths_time(initial_station, final_station)
                            time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                            if config_flags.include_scaling:
                                wait_time_inside_bus = (1/simulator_config.bus_capacities[bus_index])*(time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                            else:
                                wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                            requests_wait_time += wait_time_inside_bus
                            del(serviced_requests[dropoff_request_index])
                        else:
                            new_dropoff_index = -1 * predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["dropoff"][list_index]
                            if dropoff_request_index in request_assignment[bus_index]:
                                del(request_assignment[bus_index][dropoff_request_index])
                            predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["dropoff"][list_index] = new_dropoff_index
                
                proper_pickup_elements = []
                for pickup_request_index in pickup_requests_list:
                    if pickup_request_index != -1:
                        proper_pickup_elements.append(pickup_request_index)
                
                if len(proper_pickup_elements) == 0:
                    predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["pickup"] = [-1]
                else:
                    predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["pickup"] = proper_pickup_elements

                proper_dropoff_elements = []
                for dropoff_request_index in dropoff_requests_list:
                    if dropoff_request_index != -1:
                        proper_dropoff_elements.append(dropoff_request_index)
                
                if len(proper_dropoff_elements) == 0:
                    predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["dropoff"] = [-1]
                else:
                    predicted_bus_fleet.routing_plans[bus_index].stops_request_pairing.data[i]["dropoff"] = proper_dropoff_elements
                
                
                current_edge_cost = map_graph.obtain_shortest_paths_time(current_location, next_location)
                stop_wait_time = predicted_bus_fleet.routing_plans[bus_index].stops_wait_times[i]
                real_time += (stop_wait_time + current_edge_cost)
                route_time += (stop_wait_time + current_edge_cost)

            if config_flags.consider_route_time:
                bus_assignment_cost = route_time + requests_wait_time
            else:
                bus_assignment_cost = requests_wait_time
            
            new_assignment_cost.append(bus_assignment_cost)
        
        return new_assignment_cost
    
    def assign_requests_and_create_routes(self, state_object: State, requests, config_flags: Config_flags):
        self.dynamic_route_creation.assign_requests_and_create_routes(state_object=state_object, 
                                                                      requests=requests, 
                                                                      config_flags=config_flags)

    
