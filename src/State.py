"""State.py

This module contains all the required methods to initialize and update the state of the routing system. The state keeps track
of the variables described in the paper, and it is used to describe the locations of requests, buses, and the time left in the time
horizon.

"""
import copy

from Data_structures import Bus_fleet, Bus_stop_request_pairings, Routing_plan, Simulator_config, Date_operational_range

class State:
    def __init__(self, map_graph, date_operational_range: Date_operational_range, simulator_config: Simulator_config):
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.bus_capacities = simulator_config.bus_capacities
        self.num_buses = simulator_config.num_buses

        self.date_operational_range = date_operational_range
        self.time_horizon = (((date_operational_range.end_hour-date_operational_range.start_hour)+1) * 60) * 60
        self.bus_locations = copy.deepcopy(simulator_config.initial_bus_locations)

        self._set_default_values_for_tracking_variables()

        routing_plans = self._initialize_buses_with_default_values()
        self.bus_fleet = Bus_fleet(routing_plans=routing_plans)
    
    def _set_default_values_for_tracking_variables(self):
        self.time_spent_at_intersection = [0] * self.num_buses
        self.wait_time_at_the_station = [0] * self.num_buses
        self.step_index = [0] * self.num_buses
        self.bus_stop_index = [0] * self.num_buses
        self.passengers_in_bus = [0] * self.num_buses
        self.route_time = [0] * self.num_buses
        self.prev_passengers = [{}] * self.num_buses
        self.request_info = {}
        self.requests_pickup_times = {}
        self.request_capacities = {}
        self.executed_routes = []
        self.stage_costs = [0]
        self.state_num = 0

    def _initialize_buses_with_default_values(self):
        self.request_assignment = []
        routing_plans = []
        for bus_index in range(self.num_buses):
            self.request_assignment.append({})
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
            
    
    def reinitialize_state(self, initial_assignment_cost: list[int], predicted_bus_fleet: Bus_fleet, initial_request_capacities,
                           requests_pickup_times, simulator_config: Simulator_config):
        self.num_buses = simulator_config.num_buses
        self.bus_capacities = simulator_config.bus_capacities
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.bus_locations = copy.deepcopy(simulator_config.initial_bus_locations)

        self._set_default_values_for_tracking_variables()

        self.stage_costs = [sum(initial_assignment_cost)]
        self.request_capacities = initial_request_capacities
        self.requests_pickup_times = requests_pickup_times
        self.bus_fleet = predicted_bus_fleet
    
    def retrieve_request_assignment(self):
        return self.request_assignment
    
    def retrieve_route_info_for_bus(self, bus_index):
        request_assignment = self.request_assignment[bus_index]
        routing_plan = self.bus_fleet[bus_index]
        return request_assignment, routing_plan
    
    def _log_picked_up_request_metrics(self, bus_index, request_index):
        request_row = self.request_assignment[bus_index][request_index]
        pick_up_time = self.state_num
        desired_pickup_time = ((((request_row["Requested Pickup Time"].hour - self.date_operational_range.start_hour) * 60) \
                                + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
        pickup_deviation = pick_up_time - desired_pickup_time

        self.request_info[request_index] = {}
        self.request_info[request_index]["Wait Time"] = pickup_deviation
        self.request_info[request_index]["Number of Passengers"] = request_row["Number of Passengers"]

        if pickup_deviation < 0:
            print("Pickup Times for simulation")
            print(pick_up_time)
            print(desired_pickup_time)
            print(request_index)
            print(bus_index)


    def _log_dropped_off_request_metrics(self, bus_index: int, request_index: int):
        request_row = self.request_assignment[bus_index][request_index]
        drop_off_time = self.state_num
        pickup_deviation = self.request_info[request_index]["Wait Time"]
        desired_pickup_time = ((((request_row["Requested Pickup Time"].hour - self.date_operational_range.start_hour) * 60) \
                                + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
        pickup_time = pickup_deviation + desired_pickup_time

        time_spent_on_bus = drop_off_time - pickup_time
        request_origin = request_row["Origin Node"]
        request_destination = request_row["Destination Node"]
        direct_route_time = self.map_graph.obtain_shortest_paths_time(request_origin, request_destination)

        self.request_info[request_index]["Time Spent on Bus"] = time_spent_on_bus - direct_route_time
        self.request_info[request_index]["Trip Time"] = time_spent_on_bus

        if time_spent_on_bus - direct_route_time < 0:
            print("Time in bus for simulation")
            print(time_spent_on_bus)
            print(drop_off_time)
            print(request_index)
            print(bus_index)

    def _dropoff_passengers(self, bus_index: int, next_location: int):
        next_bus_stop_index = self.bus_stop_index[bus_index] + 1
        next_bus_stop = self.bus_fleet.routing_plans[bus_index].bus_stops[next_bus_stop_index]

        if next_bus_stop == next_location:
            self.bus_stop_index[bus_index] += 1
            request_index_list = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[self.bus_stop_index[bus_index]]["dropoff"]
            
            for request_index in request_index_list:
                if request_index >= 0:
                    if request_index in self.prev_passengers[bus_index]:
                        del(self.prev_passengers[bus_index][request_index])
                        self.passengers_in_bus[bus_index] -= self.request_capacities[request_index]
                        self._log_dropped_off_request_metrics(bus_index=bus_index, request_index=request_index)
    
    def _pickup_passengers(self, bus_index: int):
        current_bus_stop_index = self.bus_stop_index[bus_index]
        current_bus_location = self.bus_locations[bus_index]
        current_bus_stop = self.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index]

        if current_bus_location == current_bus_stop:
            request_index_list = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_bus_stop_index]["pickup"]

            for request_index in request_index_list:
                if request_index >= 0:
                    if request_index not in self.prev_passengers[bus_index]:
                        self.prev_passengers[bus_index][request_index] = [self.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index], self.state_num]
                        self.passengers_in_bus[bus_index] += self.request_capacities[request_index]
                        self._log_picked_up_request_metrics(bus_index=bus_index, 
                                                            request_index=request_index)
    
    def _terminate_route(self, bus_index: int):
        print("Bus #" + str(bus_index) + " has finished its route")
        print(self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data)
        self.executed_routes.append((self.bus_fleet.routing_plans[bus_index].route, self.bus_fleet.routing_plans[bus_index].route_edge_time, self.bus_fleet.routing_plans[bus_index].route_stop_wait_time))
        self.step_index[bus_index] = 0
        self.bus_stop_index[bus_index] = 0
        self.bus_locations[bus_index] = self.initial_bus_locations[bus_index]
        self.time_spent_at_intersection[bus_index] = 0
        self.wait_time_at_the_station[bus_index] = 0

        new_routing_plan = Routing_plan(bus_stops=[self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]],
                                        stops_wait_times=[0, 0],
                                        stops_request_pairing=Bus_stop_request_pairings([{"pickup": [-1], "dropoff": [-1]}, {"pickup": [-1], "dropoff": [-1]}]),
                                        assignment_cost=0,
                                        start_time=self.state_num,
                                        route=[self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]],
                                        route_edge_times=[0],
                                        route_stop_wait_time=[0, 0])

        self.bus_fleet.routing_plans[bus_index] = new_routing_plan
    
    def _check_traversal_times(self, bus_index: int):
        current_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[self.step_index[bus_index]]
        current_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[self.step_index[bus_index]]
        
        if self.wait_time_at_the_station[bus_index] < current_stop_wait_time:
            self.wait_time_at_the_station[bus_index] += 1
        else:
            if self.time_spent_at_intersection[bus_index] == 0:
                self.time_spent_at_intersection[bus_index] += 1
                self._pickup_passengers(bus_index=bus_index)
            elif self.time_spent_at_intersection[bus_index] < current_edge_time:
                self.time_spent_at_intersection[bus_index] += 1
            else:
                self.step_index[bus_index] += 1
                next_location = self.bus_fleet.routing_plans[bus_index].route[self.step_index[bus_index]]
                self.bus_locations[bus_index] = next_location
                self.bus_fleet.routing_plans[bus_index].start_time = copy.deepcopy(self.state_num)
                self.time_spent_at_intersection[bus_index] = 0
                self.wait_time_at_the_station[bus_index] = 0
    
                self._dropoff_passengers(bus_index=bus_index, next_location=next_location)
                if self.step_index[bus_index] == len(self.bus_fleet.routing_plans[bus_index].route) - 1:
                    self._terminate_route(bus_index=bus_index)
                else:
                    self._check_traversal_times(bus_index=bus_index)


    def _advance_along_route(self, bus_index: int):
        if self.state_num >= self.bus_fleet.routing_plans[bus_index].start_time:
            if len(self.bus_fleet.routing_plans[bus_index].route) > 2:
                self._check_traversal_times(bus_index=bus_index)
                self.route_time[bus_index] += 1
            else:
                self.bus_fleet.routing_plans[bus_index].start_time = self.state_num + 1

    def update_state(self, bus_index: int, request_index: int, request_row, assignment_cost: int, new_routing_plan: Routing_plan):
        self.stage_costs[self.state_num] += assignment_cost

        if len(self.bus_fleet.routing_plans[bus_index].bus_stops) == 2 and \
            self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index]] == self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index]+1]:
            self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index]] + new_routing_plan.bus_stops
            self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_wait_times
            self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_request_pairing.data
        else:
            if new_routing_plan.route[0] == self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index] + 1]:
                self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.bus_stops
                self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.stops_wait_times
                self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.stops_request_pairing.data
            else:
                self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index]] + new_routing_plan.bus_stops
                self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_wait_times
                self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_request_pairing.data
        
        if len(self.bus_fleet.routing_plans[bus_index].route) > 2:
            current_step_index = self.step_index[bus_index]
            current_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]

            if self.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                self.bus_fleet.routing_plans[bus_index].route = self.bus_fleet.routing_plans[bus_index].route[:self.step_index[bus_index]] + new_routing_plan.route
                self.bus_fleet.routing_plans[bus_index].route_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[:self.step_index[bus_index]] + new_routing_plan.route_edge_time
                self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[:self.step_index[bus_index]] + new_routing_plan.route_stop_wait_time
            else:
                self.bus_fleet.routing_plans[bus_index].route = self.bus_fleet.routing_plans[bus_index].route[:self.step_index[bus_index]+1] + new_routing_plan.route
                self.bus_fleet.routing_plans[bus_index].route_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[:self.step_index[bus_index]+1] + new_routing_plan.route_edge_time
                self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[:self.step_index[bus_index]+1] + new_routing_plan.route_stop_wait_time
        else:
            self.bus_fleet.routing_plans[bus_index].route = new_routing_plan.route
            self.bus_fleet.routing_plans[bus_index].route_edge_time= new_routing_plan.route_edge_time
            self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = new_routing_plan.route_stop_wait_time
        self.request_assignment[bus_index][request_index] = request_row
    
    def next_state(self):
        for j in range(len(self.bus_locations)):
            self._advance_along_route(j)
        
        self.stage_costs.append(0)
        self.state_num += 1
    
    def calculate_average_route_time(self):
        route_time_list = []

        for executed_route_tuple in self.executed_routes:
            route_times = executed_route_tuple[1]
            stop_wait_times_list = executed_route_tuple[2]
            route_time = sum(route_times)
            stop_wait_times = sum(stop_wait_times_list)
            route_time_list.append(route_time+stop_wait_times)

        avg_route_time = sum(route_time_list)/(len(route_time_list))
        return avg_route_time
    
    def calculate_total_route_time(self):
        route_time_list = []

        for executed_route_tuple in self.executed_routes:
            route_times = executed_route_tuple[1]
            stop_wait_times_list = executed_route_tuple[2]
            route_time = sum(route_times)
            stop_wait_times = sum(stop_wait_times_list)
            route_time_list.append(route_time+stop_wait_times)

        total_route_time = sum(route_time_list)
        return total_route_time
    
    def calculate_average_passenger_wait_time(self):
        wait_times_at_station = []
        wait_times_on_bus = []
        trip_times = []
        number_of_passengers_list = []
        for request_key in self.request_info.keys():
            wait_time_on_bus = self.request_info[request_key]["Time Spent on Bus"]
            passenger_trip_time = self.request_info[request_key]["Trip Time"]
            wait_time_at_station = self.request_info[request_key]["Wait Time"]
            number_of_passengers = self.request_info[request_key]["Number of Passengers"]
            number_of_passengers_list.append(number_of_passengers)
            wait_times_at_station.append(wait_time_at_station * number_of_passengers)
            wait_times_on_bus.append(wait_time_on_bus * number_of_passengers)
            trip_times.append(passenger_trip_time * number_of_passengers)
        
        avg_wait_time_at_station = sum(wait_times_at_station)/sum(number_of_passengers_list)
        avg_wait_time_on_bus = sum(wait_times_on_bus)/sum(number_of_passengers_list)
        avg_trip_time = sum(trip_times)/sum(number_of_passengers_list)

        number_of_requests = len(wait_times_at_station)
        number_of_passengers = sum(number_of_passengers_list)

        return avg_wait_time_at_station, avg_wait_time_on_bus, avg_trip_time, number_of_requests, number_of_passengers
    
    def calculate_requests_wait_times(self):
        wait_times_at_station = []
        wait_times_on_bus = []
        trip_times = []
        for request_key in self.request_info.keys():
            wait_time_on_bus = self.request_info[request_key]["Time Spent on Bus"]
            wait_time_at_station = self.request_info[request_key]["Wait Time"]
            passenger_trip_time = self.request_info[request_key]["Trip Time"]
            wait_times_at_station.append(wait_time_at_station)
            wait_times_on_bus.append(wait_time_on_bus)
            trip_times.append(passenger_trip_time)

        return wait_times_at_station, wait_times_on_bus, trip_times
    
    def calculate_total_cost(self, consider_route_time):
        if consider_route_time:
            route_time = self.calculate_total_route_time()
        else:
            route_time = 0
        
        wait_times_at_station = []
        wait_times_on_bus = []
        for request_key in self.request_info.keys():
            wait_time_on_bus = self.request_info[request_key]["Time Spent on Bus"]
            wait_time_at_station = self.request_info[request_key]["Wait Time"]
            number_of_passengers = self.request_info[request_key]["Number of Passengers"]
            wait_times_at_station.append(wait_time_at_station * number_of_passengers)
            wait_times_on_bus.append(wait_time_on_bus * number_of_passengers)
        
        total_wait_time_at_station = sum(wait_times_at_station)
        total_wait_time_on_bus = sum(wait_times_on_bus)

        total_cost = route_time + total_wait_time_on_bus + total_wait_time_at_station

        return total_cost