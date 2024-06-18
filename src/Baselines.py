"""Policies.py

This module has the classes that define all routing baselines that we compare in the paper.

"""
import os
import copy
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from Insertion_procedure import Request_Insertion_Procedure, Request_Insertion_Procedure_greedy_MCTS
from Requests_predictor import Request_Prediction_Handler
from Plot_utils import Plot_utils

from Data_structures import Bus_stop_request_pairings, Config_flags, Data_folders, Dataframe_row, Requests_info, Simulator_config, Date_operational_range, Bus_fleet, Routing_plan
from State import State

import copy
from functools import partial
import pandas as pd

from State import State
from Insertion_procedure import Request_Insertion_Procedure
from Requests_predictor import Request_Prediction_Handler
from Plot_utils import Plot_utils
from Request_handler import Request_handler

from benchmark_1.MCTS import MCForest
from benchmark_1.algo1 import Actions
from benchmark_1.generative_model import GenerativeModel, Memory
# from benchmark_1.MCTS import MCForest
from benchmark_1.RV_Graph import RVGraph
from benchmark_1.VV_Graph import VVGraph
# from benchmark_1.algo1 import Actions
from benchmark_1.new_DS import Buses, Request_Insertion_Procedure_baseline_1, SimRequest, Route
import benchmark_1.config as config 

from typing import Callable


class Greedy_static_insertion:
    
    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, perfect_accuracy: bool = True):
        
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.bus_capacities = simulator_config.bus_capacities
        self.num_buses = simulator_config.num_buses
        new_results_folder = os.path.join(data_folders.static_results_folder, "greedy")
        if not os.path.isdir(new_results_folder):
            os.mkdir(new_results_folder)
        self.results_folder = new_results_folder
        self.total_cost = 0
        self.req_pred_handler = Request_Prediction_Handler(data_folders=data_folders,
                                                           perfect_accuracy=perfect_accuracy)
        self.req_insert = Request_Insertion_Procedure(map_graph=map_graph)
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
    
    def retrieve_route_info_for_bus(self, bus_index) -> Routing_plan:
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
    
    def _insert_requests(self, requests_df, requests_info: Requests_info, config_flags: Config_flags):

        for index, row, in requests_df.iterrows():
            min_assignment_cost = float("inf")
            min_bus_index = 0
            min_routing_plan = None
            for bus_index in range(self.num_buses):
                local_routing_plan = self.bus_fleet.routing_plans[bus_index]
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
            
            if config_flags.verbose:
                print("Cost for assigning request " + str(index) + " to bus " + str(min_bus_index) + " = " + str(min_assignment_cost))

            
            self.request_assignment[min_bus_index][index] = row
            self.current_assignment[min_bus_index] = [index]

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
        
        
    def assign_requests_and_create_routes(self, date_operational_range: Date_operational_range, config_flags: Config_flags):
        if config_flags.plot_initial_routes:
            self.plot_utils.reset_frame_number()

        requests_to_be_serviced = self._extract_requests(date_operational_range=date_operational_range)

        requests_info = Requests_info(requests_df=requests_to_be_serviced, start_hour=date_operational_range.start_hour)

        self._insert_requests(requests_df=requests_to_be_serviced,
                              requests_info=requests_info,
                              config_flags=config_flags)

        assignment_costs = []

        for routing_plan in self.bus_fleet.routing_plans:
            current_assignment_cost = routing_plan.assignment_cost
            assignment_costs.append(current_assignment_cost)

        self.total_cost = sum(assignment_costs)

        return self.total_cost, assignment_costs, requests_info


class Greedy_dynamic_insertion:

    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags):
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
        self.include_scaling = config_flags.include_scaling
    
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
            assignment_result = self._determine_assignment(state_object=state_object,
                                                           current_request_index=request_index,
                                                           current_request_row=request_row, 
                                                           config_flags=config_flags)
            bus_index, assignment_cost, new_routing_plan = assignment_result

            current_step_index = state_object.step_index[bus_index]
            current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
            if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]]
            else:
                current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]+1]

            route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                bus_stops=new_routing_plan.bus_stops, 
                                                                                                stops_wait_times=new_routing_plan.stops_wait_times)
            
            new_routing_plan.update_routes(route=route,
                                           route_edge_times=bus_route_edge_time,
                                           route_stop_wait_time=routes_stop_wait_time)
            
            state_object.update_state(bus_index, 
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


class MCTS:

    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags, cost_func: str, request_handler: Request_handler) -> None:
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure_greedy_MCTS(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
        self.include_scaling = config_flags.include_scaling
        self.cost_func = cost_func
        self.request_handler = request_handler
        self.gen = None
    
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

    def _get_dynamic_insertion_cost_for_request(self, state_object: State, bus_index: int, request_index: int, 
                                                request_row, config_flags: Config_flags, cost_func: str, plans: list[Routing_plan], current_bus_index: int, is_full: bool,
                                                is_for_RV: bool = False, is_for_VV: bool = True):
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
                                                            stops_sequence=plans[bus_index].bus_stops, 
                                                            stops_wait_time=plans[bus_index].stops_wait_times, 
                                                            stop_request_pairing=plans[bus_index].stops_request_pairing.data,
                                                            request_capacities=state_object.request_capacities,
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination, 
                                                            requests_pickup_times=state_object.requests_pickup_times,
                                                            request_index=request_index,
                                                            time_horizon = state_object.time_horizon,
                                                            state_num = state_object.state_num,
                                                            cost_func = cost_func,
                                                            consider_wait_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling,
                                                            current_bus_index = bus_index,
                                                            is_full = is_full)
                                                            
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, rv_list = insertion_result

        # new_assignment_cost = state_object.bus_fleet.routing_plans[bus_index].assignment_cost + total_dev_cost
        new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)

        new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                        stops_wait_times=full_stops_wait_time,
                                        stops_request_pairing=new_stop_req_pairings,
                                        assignment_cost=total_dev_cost,
                                        start_time=state_object.bus_fleet.routing_plans[bus_index].start_time,
                                        route=[],
                                        route_edge_times=[],
                                        route_stop_wait_time=[])

        
        return total_dev_cost, new_routing_plan, rv_list

    #TODO change manualy overriden num of buses
    def _determine_assignment(self, state_object, current_request_index, current_request_row, config_flags: Config_flags, cost_func: str,
                              plans: list[Routing_plan], is_full: bool, num_buses = 3, is_for_RV: bool = False, is_for_VV: bool = True):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        all_possible_rv_paths: list[tuple[int, Routing_plan]] = []

        for bus_index in range(config.NUM_BUSSES):
            total_dev_cost, new_routing_plan, rv_list = self._get_dynamic_insertion_cost_for_request(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            request_index=current_request_index,
                                                                                            request_row=current_request_row,
                                                                                            config_flags=config_flags,
                                                                                            cost_func = cost_func,
                                                                                            plans = plans,
                                                                                            current_bus_index = bus_index,
                                                                                            is_full= is_full)
            all_possible_rv_paths.extend(rv_list)

            if total_dev_cost < min_assignment_cost:
                min_assignment_cost = total_dev_cost
                min_bus_index = bus_index
                min_routing_plan = new_routing_plan
        
        return min_bus_index, min_assignment_cost, min_routing_plan, all_possible_rv_paths
    
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
        routes_stop_wait_time += [stops_wait_times[-1]]
        
        return bus_route, bus_route_edge_time, routes_stop_wait_time

    def assign_requests_and_create_routes(self, state_object: State, requests, config_flags: Config_flags):
        
        #TODO turn off back again once the debuggin is done!
        if self.gen is None:
            self.gen = GenerativeModel(state_object, self.request_handler)
            #TODO rewrite this to not bloat the state!
            state_object.request_capacities |= self.gen.historic_requests_capacities
            state_object.requests_pickup_times |= self.gen.historic_requests_pickup_times #TODO these must be changed as some of them may exist in the future

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

            request = SimRequest(request_row['Origin Node'],
                                 request_row['Destination Node'],
                                 request_index)
            
            initial_routing_plans: list[Routing_plan] = copy.deepcopy(state_object.bus_fleet.routing_plans) #SLICE AND PASS SLICED!

            current_start_time: list[int] = [] 
            current_stop_index: list[int] = [] 
            current_location: list[int] = [] 
            passengers_in_bus: list[int] = [] 
            prev_passengers: list[int] = [] #TODO waht is type of this?

            for bus_index in range(state_object.num_buses):
                out = self._get_bus_parameters_of_interest(state_object=state_object, bus_index=bus_index)
                
                current_start_time.append(out[0])
                current_stop_index.append(out[1])
                stop_index = out[1]
                current_location.append(out[2])
                passengers_in_bus.append(out[3])
                prev_passengers.append(out[4])
                
                #INITIAL ROUTING PLANS ARE SLICED
                initial_routing_plans[bus_index].bus_stops = initial_routing_plans[bus_index].bus_stops[stop_index:]
                initial_routing_plans[bus_index].stops_wait_times = initial_routing_plans[bus_index].stops_wait_times[stop_index:]
                initial_routing_plans[bus_index].stops_request_pairing = Bus_stop_request_pairings(initial_routing_plans[bus_index].stops_request_pairing.data[stop_index:])

            #TODO this must be called on the partial list!
            greedy_assignment_rv:  Callable[[list[Routing_plan]], tuple[int, int, Routing_plan, list[int]]]
            greedy_assignment_rv = partial(self._determine_assignment, 
                                           state_object=state_object, 
                                           config_flags=config_flags, 
                                           cost_func=self.cost_func,
                                           is_full=False)  #TODO enable this on the first run!

            #TODO this must be called on the partial list!
            allocate_method_vv:  Callable[[SimRequest, list[Routing_plan]], tuple[int, Routing_plan, list[int]]]
            allocate_method_vv = partial(self._get_dynamic_insertion_cost_for_request,
                                         state_object = state_object, 
                                         config_flags = config_flags, 
                                         cost_func = self.cost_func,
                                         is_full=False) #pass this with full parameters

            unallocate_method_vv:  Callable[[int, Routing_plan, int], tuple[Routing_plan, SimRequest]]
            unallocate_method_vv = partial(self.req_insert.unallocate,
                                           requests_pickup_times = state_object.requests_pickup_times,
                                           requests_capacities = state_object.request_capacities,
                                           time_horizon = state_object.time_horizon, 
                                           state_num = state_object.state_num,
                                           current_start_time = current_start_time, 
                                           current_stop_index = current_stop_index, 
                                           current_location = current_location, 
                                           passengers_in_bus = passengers_in_bus, 
                                           prev_passengers = prev_passengers) #TODO add passable cost function here!

            out = MCForest(initial_routing_plans = initial_routing_plans, 
                           initial_request = request, 
                           allocate_method_vv = allocate_method_vv, 
                           unallocate_method_vv = unallocate_method_vv, 
                           current_stop_index = current_stop_index, 
                           greedy_assignment_rv = greedy_assignment_rv, 
                           generative_model = self.gen).get_best_action()

            for bus_index, new_routing_plan in enumerate(out): 
            
                current_step_index = state_object.step_index[bus_index]
                current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
                if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]]
                else:
                    current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]+1]

                route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                    bus_stops=new_routing_plan.bus_stops, 
                                                                                                    stops_wait_times=new_routing_plan.stops_wait_times)
                
                new_routing_plan.update_routes(route=route,
                                            route_edge_times=bus_route_edge_time,
                                            route_stop_wait_time=routes_stop_wait_time)
                
                state_object.update_state(bus_index, 
                                        request_index=request_index,
                                        request_row=request_row,
                                        assignment_cost=new_routing_plan.assignment_cost,
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


class Greedy_dynamic_insertion_MCTS_baseline:

    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags, cost_func: str):
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure_greedy_MCTS(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
        self.include_scaling = config_flags.include_scaling
        self.cost_func = cost_func
    
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

    def _get_dynamic_insertion_cost_for_request(self, state_object: State, bus_index: int, request_index: int, request_row, config_flags: Config_flags, cost_func: str):
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
                                                            time_horizon = state_object.time_horizon,
                                                            state_num = state_object.state_num,
                                                            cost_func = cost_func,
                                                            consider_wait_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling,
                                                            current_bus_index = 0,
                                                            is_full = True)
                                                            
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _ = insertion_result

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

    def _determine_assignment(self, state_object, current_request_index, current_request_row, config_flags: Config_flags, cost_func: str):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        for bus_index in range(self.num_buses):
            total_dev_cost, new_routing_plan = self._get_dynamic_insertion_cost_for_request(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            request_index=current_request_index,
                                                                                            request_row=current_request_row,
                                                                                            config_flags=config_flags,
                                                                                            cost_func = cost_func)

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
        routes_stop_wait_time += [stops_wait_times[-1]]
        
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
            assignment_result = self._determine_assignment(state_object=state_object,
                                                           current_request_index=request_index,
                                                           current_request_row=request_row, 
                                                           config_flags=config_flags,
                                                           cost_func = self.cost_func)
            bus_index, assignment_cost, new_routing_plan = assignment_result

            current_step_index = state_object.step_index[bus_index]
            current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
            if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]]
            else:
                current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]+1]

            route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                bus_stops=new_routing_plan.bus_stops, 
                                                                                                stops_wait_times=new_routing_plan.stops_wait_times)
            
            new_routing_plan.update_routes(route=route,
                                           route_edge_times=bus_route_edge_time,
                                           route_stop_wait_time=routes_stop_wait_time)
            
            state_object.update_state(bus_index, 
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

class Predictive_insertion:

    def __init__(self):
        pass