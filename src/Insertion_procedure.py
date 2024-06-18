import copy
from typing import Callable

from Data_structures import Bus_stop_request_pairings, Routing_plan
from benchmark_1.new_DS import SimRequest

class Request_Insertion_Procedure:
    def __init__(self, map_graph):
        self.map_graph = map_graph
    
    def _calculate_cost_of_route(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                 requests_pickup_times, request_capacities, prev_passengers, consider_route_time=False, include_scaling=False, 
                                 maximize=False, bus_capacity=20):
        route_time = 0
        requests_wait_time = 0
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        print("Pickup Times")
                        print("Actual pickup time = " + str(actual_pickup_time))
                        print("Desired pickup time = " + str(request_desired_pickup_time))
                        print(i)
                        print(current_start_time)
                        print(pickup_request_index)
                        print(stops_request_pair)
                        print(stops_wait_time)
                    wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                    requests_wait_time += wait_time_at_the_station

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
            
            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            print(direct_route_time)
                            print("Time in bus")
                            print(time_in_bus_route)
                            print(direct_route_time)
                            print("Real time = " + str(real_time))
                            print("Start Time = " + str(current_start_time))
                            print("Request index = " + str(dropoff_request_index))
                            print("Actual pickup time = " + str(serviced_requests[dropoff_request_index][1]))
                            print(stops_sequence)
                            print(stops_wait_time)
                            print(stops_request_pair)

                        wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                        requests_wait_time += wait_time_inside_bus
                        del(serviced_requests[dropoff_request_index])
            
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)
            route_time += (stop_wait_time + current_edge_cost)

        if consider_route_time:
            bus_assignment_cost = route_time + requests_wait_time
        else:
            bus_assignment_cost = requests_wait_time
        
        return bus_assignment_cost
    
    def _place_request_inside_stop(self, local_stop_request_pairings, stop_index, request_index, label):
        if local_stop_request_pairings[stop_index][label][0] == -1:
            local_stop_request_pairings[stop_index][label][0] = request_index
        else:
            request_placed = False
            for current_list_index, current_request_index in enumerate(local_stop_request_pairings[stop_index][label]):
                if current_request_index == -1*request_index:
                    local_stop_request_pairings[stop_index][label][current_list_index] = request_index
                    request_placed = True
                    break

            if not request_placed:
                local_stop_request_pairings[stop_index][label].append(request_index)
    
    def _update_stop_request_pairings(self, stop_request_pairings, stop_index, request_index, pickup=False, insert=False):
        if insert:
            if pickup:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [request_index], "dropoff": [-1]}] + stop_request_pairings[stop_index:]
            else:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [-1], "dropoff": [request_index]}] + stop_request_pairings[stop_index:]
        else:
            local_stop_request_pairings = copy.deepcopy(stop_request_pairings)
            if pickup:
                label = "pickup"
            else:
                label = "dropoff"
            self._place_request_inside_stop(local_stop_request_pairings=local_stop_request_pairings,
                                            stop_index=stop_index,
                                            request_index=request_index,
                                            label=label)
        
        return local_stop_request_pairings
    
    def _update_stop_wait_times(self, local_travel_time, stop_index, stops_sequence, stop_request_pairings, stops_wait_time, requests_pickup_times,
                                default_stop_wait_time=15):
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(stop_index, len(new_stops_wait_time)-1):
            current_request_index_dict = stop_request_pairings[i]
            pickup_requests_list = current_request_index_dict["pickup"]

            for list_index, current_request_index in enumerate(pickup_requests_list):
                if current_request_index == -1:
                    continue
                else:
                    if list_index == 0:
                        new_stops_wait_time[i] = default_stop_wait_time
                    current_request_pickup_time = requests_pickup_times[current_request_index]
                    new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
            current_location = stops_sequence[i]
            wait_time = new_stops_wait_time[i]
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            local_travel_time += (wait_time + current_edge_cost)
        
        return new_stops_wait_time
    
    def _create_new_stop_lists(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, 
                               stop_request_pairings, pickup=False, default_stop_wait_time=60):
        if request_node == stops_sequence[next_index]:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index-1
            insert_flag = False

        else:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index

    def _create_new_stop_lists_online(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, bus_location,
                               stop_request_pairings, pickup=False, default_stop_wait_time=60, mismatched_flag=False):
        
        if request_node == stops_sequence[next_index]:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            if mismatched_flag:
                local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
                local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
                insertion_index = next_index
                insert_flag = True
            else:
                local_stops_sequence = stops_sequence
                local_stops_wait_time = stops_wait_time
                insertion_index = next_index-1
                insert_flag = False

        else:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index
    
    def _insert_pickup_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        if total_travel_time == 0:
            time_until_request_available = max(0, requests_pickup_times[request_index]-time_to_pickup)
            new_start_time = time_until_request_available
            current_stop_wait_time = default_stop_wait_time
        else:
            new_start_time = current_start_time
            current_travel_time = total_travel_time + current_start_time + time_to_pickup + stops_wait_time[next_index - 1]
            current_stop_wait_time = max(default_stop_wait_time, (requests_pickup_times[request_index] - current_travel_time) + default_stop_wait_time)
        
        new_travel_time = total_travel_time + new_start_time + time_to_pickup

        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, new_start_time, insertion_index
    
    def _insert_pickup_in_route_online(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60, mismatched_flag=False):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        new_travel_time = total_travel_time + current_start_time + time_to_pickup
        current_stop_wait_time = max(default_stop_wait_time, (requests_pickup_times[request_index] - new_travel_time) + default_stop_wait_time)

        new_lists = self._create_new_stop_lists_online(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                bus_location=current_location,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time,
                                                mismatched_flag=mismatched_flag)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, current_start_time, insertion_index
    
    def _insert_dropoff_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                next_index, request_destination, requests_pickup_times, stop_request_pairings, request_index, 
                                default_dropoff_wait_time=20, default_stop_wait_time=60):
        
        time_to_dropoff = self.map_graph.obtain_shortest_paths_time(current_location, request_destination)
        
        current_stop_wait_time = default_dropoff_wait_time

        new_travel_time = total_travel_time + current_start_time + time_to_dropoff
        
        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_destination,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=False,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, _ = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion
    
    def _obtain_passengers_in_bus(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _obtain_passengers_in_bus_online(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _place_request_offline_exact(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, request_origin, request_destination, requests_pickup_times, 
                                     stop_request_pairings, request_index, request_capacities, consider_route_time=True, include_scaling=False):
        total_travel_time = 0
        min_cost = float("inf")
        min_start_time = 0
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        serviced_requests = {}
        passenger_in_bus = 0
        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=stops_sequence, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=stops_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity)
        for i in range(len(stops_sequence)-1):
            passenger_in_bus = self._obtain_passengers_in_bus(stop_index=i, 
                                                            travel_time=total_travel_time+current_start_time,
                                                            bus_stops=stops_sequence,
                                                            stops_wait_time=stops_wait_time,
                                                            passenger_in_bus=passenger_in_bus,
                                                            stop_request_pairings=stop_request_pairings,
                                                            serviced_requests=serviced_requests,
                                                            request_capacities=request_capacities)
            current_location = stops_sequence[i]
            next_location = stops_sequence[i+1]
            next_index = i+1

            if passenger_in_bus + request_capacities[request_index] <= bus_capacity:
                deviation_result = self._insert_pickup_in_route(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=stops_sequence,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index)
                    
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                local_passengers_in_bus = copy.deepcopy(passenger_in_bus)
                local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    new_total_travel_time = total_travel_time + stops_wait_time[i] + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]
                    
                    local_passengers_in_bus = self._obtain_passengers_in_bus(stop_index=j,
                                                                            travel_time=full_travel_time+new_start_time,
                                                                            bus_stops=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            passenger_in_bus=local_passengers_in_bus,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            serviced_requests=local_serviced_requests,
                                                                            request_capacities=request_capacities)

                    total_passengers_in_bus = local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=new_start_time,
                                                                                total_travel_time=full_travel_time,
                                                                                stops_sequence=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                current_location=new_current_location,
                                                                                next_index=new_next_index,
                                                                                request_destination=request_destination,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=new_start_time,
                                                            stops_sequence=full_stop_sequence,
                                                            stops_wait_time=full_stops_wait_time,
                                                            stops_request_pair=full_stop_req_pair,
                                                            bus_location=full_stop_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity)

                        total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair
                            
                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time += (stops_wait_time[i] + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def _place_request_online_exact(self, current_start_time, bus_capacity, bus_location, planned_stops, stops_wait_time, request_origin, 
                                    request_destination, requests_pickup_times, stop_request_pairings, passengers_in_bus, 
                                    prev_passengers, request_index, request_capacities, consider_route_time=True, include_scaling=False,):
        total_travel_time = 0
        min_cost = float("inf")
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        min_start_time = 0
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        
        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=planned_stops, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=bus_location,
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            prev_passengers=prev_passengers,
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity)
        
        for i in range(len(planned_stops)-1):
            if i == 0 and bus_location != planned_stops[0]:
                local_passengers_in_bus = local_passengers_in_bus
                current_location = bus_location
            else:
                local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                                        travel_time=total_travel_time+current_start_time,
                                                                        bus_stops=planned_stops,
                                                                        stops_wait_time=stops_wait_time,
                                                                        passenger_in_bus=local_passengers_in_bus,
                                                                        stop_request_pairings=stop_request_pairings,
                                                                        serviced_requests=serviced_requests,
                                                                        request_capacities=request_capacities)
                current_location = planned_stops[i]

            next_location = planned_stops[i+1]
            next_index = i+1
            
            if local_passengers_in_bus  + request_capacities[request_index] <= bus_capacity:
                if i == 0 and bus_location != planned_stops[0]:
                    mismatched_flag = True
                else:
                    mismatched_flag = False
                
                deviation_result = self._insert_pickup_in_route_online(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=planned_stops,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index,
                                                                mismatched_flag=mismatched_flag)
                    
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                new_local_passengers_in_bus = copy.deepcopy(local_passengers_in_bus)
                new_local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    if i == 0 and bus_location != planned_stops[0]:
                        stop_time = 0
                    else:
                        stop_time = stops_wait_time[i]
                    new_total_travel_time = total_travel_time + stop_time + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    new_full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]

                    new_local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=j,
                                                                                travel_time=new_full_travel_time+current_start_time,
                                                                                bus_stops=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                passenger_in_bus=new_local_passengers_in_bus,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                serviced_requests=new_local_serviced_requests,
                                                                                request_capacities=request_capacities)

                    total_passengers_in_bus = new_local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=current_start_time,
                                                                            total_travel_time=new_full_travel_time,
                                                                            stops_sequence=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            current_location=new_current_location,
                                                                            next_index=new_next_index,
                                                                            request_destination=request_destination,
                                                                            requests_pickup_times=requests_pickup_times,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                                        stops_sequence=full_stop_sequence, 
                                                                        stops_wait_time=full_stops_wait_time,
                                                                        stops_request_pair=full_stop_req_pair,
                                                                        bus_location=bus_location,
                                                                        requests_pickup_times=requests_pickup_times,
                                                                        request_capacities=request_capacities, 
                                                                        prev_passengers=prev_passengers,
                                                                        consider_route_time=consider_route_time,
                                                                        include_scaling=include_scaling,
                                                                        bus_capacity=bus_capacity)

                        total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < 0:
                            print("Request index = " + str(request_index))
                            print("New bus stops = " + str(full_stop_sequence))


                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair

                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            if i == 0 and bus_location != planned_stops[0]:
                current_wait_time = 0
            else:
                current_wait_time = stops_wait_time[i]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def static_insertion(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, stop_request_pairing, requests_pickup_times, request_capacities, 
                         request_origin, request_destination, request_index, consider_route_time=True, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence)
        local_stops_wait_time = copy.deepcopy(stops_wait_time)
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing)

        deviation_result = self._place_request_offline_exact(current_start_time=current_start_time,
                                                             bus_capacity=bus_capacity,
                                                             stops_sequence=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time
    
    def dynamic_insertion(self, current_start_time, current_stop_index, bus_capacity, passengers_in_bus, prev_passengers, bus_location,
                          stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, request_origin, request_destination, 
                          requests_pickup_times, request_index, consider_route_time=True, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
        local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])

        deviation_result = self._place_request_online_exact(current_start_time=current_start_time, 
                                                            bus_capacity=bus_capacity,
                                                            bus_location=bus_location,
                                                             planned_stops=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _ = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair
    

class Request_Insertion_Procedure_greedy_MCTS:
    def __init__(self, map_graph) -> None:
        self.map_graph = map_graph
    
    def _calculate_cost_of_route(self, stops_sequence: list[int], stops_wait_time: list[int], stops_request_pair: list[dict[str, list[int]]], 
                                 request_capacities: dict[int, int], passengers_in_bus: int, time_horizon: int, state_num: int, 
                                 current_start_time: int, prev_passengers: dict[int, list[int]], bus_location: int, 
                                 cost_func: str = 'PTT') -> int:
        route_cost = 0
        w = passengers_in_bus
        total_travel_time = 0
        cost_ptt = 0
        cost_budget = 0
        serviced_requests = copy.deepcopy(prev_passengers)
        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                current_location = bus_location
            else:
                w = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                          travel_time=total_travel_time+current_start_time,
                                                          bus_stops=stops_sequence,
                                                          stops_wait_time=stops_wait_time,
                                                          passenger_in_bus=w,
                                                          stop_request_pairings=stops_request_pair,
                                                          serviced_requests=serviced_requests,
                                                          request_capacities=request_capacities)
                                                          

                if w < 0:
                    a = 10
                assert w >= 0, 'passenger number is negative!'
                current_location = stops_sequence[i]

            next_location = stops_sequence[i+1]
            if i == 0 and bus_location != stops_sequence[0]:
                current_wait_time = 0
            else:
                current_wait_time = stops_wait_time[i]

            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

            cost_ptt += w * (current_wait_time + current_edge_cost)
            cost_budget += (1 if w > 0 else 0) * (current_wait_time + current_edge_cost)

        match cost_func:
            case 'PTT':
                route_cost = cost_ptt
                assert route_cost >= 0, 'negative cost of the route'
            case 'future requests budget':
                #TODO WORKOUT WHAT HAPPENS WHEN 
                route_cost =  2*time_horizon - current_start_time - cost_budget
                if route_cost < 0:
                    print('WRONG')
                assert route_cost >= 0, 'negative cost of the route'
                route_cost = -route_cost #TODO push negative to minimize and change from utility to costs
            case _:
                raise ValueError('You passed an undefined cost function type!')
        return route_cost

    def _place_request_inside_stop(self, local_stop_request_pairings, stop_index, request_index, label):
        if local_stop_request_pairings[stop_index][label][0] == -1:
            local_stop_request_pairings[stop_index][label][0] = request_index
        else:
            request_placed = False
            for current_list_index, current_request_index in enumerate(local_stop_request_pairings[stop_index][label]):
                if current_request_index == -1*request_index:
                    local_stop_request_pairings[stop_index][label][current_list_index] = request_index
                    request_placed = True
                    break

            if not request_placed:
                local_stop_request_pairings[stop_index][label].append(request_index)
    
    def _update_stop_request_pairings(self, stop_request_pairings, stop_index, request_index, pickup=False, insert=False):
        if insert:
            if pickup:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [request_index], "dropoff": [-1]}] + stop_request_pairings[stop_index:]
            else:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [-1], "dropoff": [request_index]}] + stop_request_pairings[stop_index:]
        else:
            local_stop_request_pairings = copy.deepcopy(stop_request_pairings)
            if pickup:
                label = "pickup"
            else:
                label = "dropoff"
            self._place_request_inside_stop(local_stop_request_pairings=local_stop_request_pairings,
                                            stop_index=stop_index,
                                            request_index=request_index,
                                            label=label)
        
        return local_stop_request_pairings
    
    def _update_stop_wait_times(self, local_travel_time, stop_index, stops_sequence, stop_request_pairings, stops_wait_time, requests_pickup_times,
                                default_stop_wait_time=60):
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(stop_index, len(new_stops_wait_time)-1):
            current_request_index_dict = stop_request_pairings[i]
            pickup_requests_list = current_request_index_dict["pickup"]

            for list_index, current_request_index in enumerate(pickup_requests_list):
                if current_request_index == -1:
                    continue
                else:
                    if list_index == 0:
                        new_stops_wait_time[i] = default_stop_wait_time
                    current_request_pickup_time = requests_pickup_times[current_request_index]
                    new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
            current_location = stops_sequence[i]
            wait_time = new_stops_wait_time[i]
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            local_travel_time += (wait_time + current_edge_cost)
        
        return new_stops_wait_time
    
    def _create_new_stop_lists(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, 
                               stop_request_pairings, pickup=False, default_stop_wait_time=60):
        if request_node == stops_sequence[next_index]:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index-1
            insert_flag = False

        else:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index

    def _create_new_stop_lists_online(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, bus_location,
                               stop_request_pairings, pickup=False, default_stop_wait_time=60, mismatched_flag=False):
        
        if request_node == stops_sequence[next_index]:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            if mismatched_flag:
                local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
                local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
                insertion_index = next_index
                insert_flag = True
            else:
                local_stops_sequence = stops_sequence
                local_stops_wait_time = stops_wait_time
                insertion_index = next_index-1
                insert_flag = False

        else:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index
    
    def _insert_pickup_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        if total_travel_time == 0:
            time_until_request_available = max(0, requests_pickup_times[request_index]-time_to_pickup)
            new_start_time = time_until_request_available
            current_stop_wait_time = default_stop_wait_time
        else:
            new_start_time = current_start_time
            current_travel_time = total_travel_time + current_start_time + time_to_pickup
            current_stop_wait_time = max(default_stop_wait_time, (requests_pickup_times[request_index] - current_travel_time) + default_stop_wait_time)
        
        new_travel_time = total_travel_time + new_start_time + time_to_pickup

        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, new_start_time, insertion_index
    
    def _insert_pickup_in_route_online(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60, mismatched_flag=False):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        new_travel_time = total_travel_time + current_start_time + time_to_pickup
        current_stop_wait_time = max(default_stop_wait_time, (requests_pickup_times[request_index] - new_travel_time) + default_stop_wait_time)

        new_lists = self._create_new_stop_lists_online(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                bus_location=current_location,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time,
                                                mismatched_flag=mismatched_flag)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, current_start_time, insertion_index
    
    def _insert_dropoff_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                next_index, request_destination, requests_pickup_times, stop_request_pairings, request_index, 
                                default_stop_wait_time=60):
        
        time_to_dropoff = self.map_graph.obtain_shortest_paths_time(current_location, request_destination)
        
        current_stop_wait_time = default_stop_wait_time

        new_travel_time = total_travel_time + current_start_time + time_to_dropoff
        
        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_destination,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=False,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, _ = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion
    
    def _obtain_passengers_in_bus(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _obtain_passengers_in_bus_online(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _place_request_offline_exact(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, request_origin, request_destination, requests_pickup_times, 
                                     stop_request_pairings, request_index, request_capacities, consider_wait_times=True, include_scaling=False):
        total_travel_time = 0
        min_cost = float("inf")
        min_start_time = 0
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        serviced_requests = {}
        passenger_in_bus = 0
        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=stops_sequence, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=stops_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            prev_passengers={},
                                                            consider_wait_time=consider_wait_times,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity)
        for i in range(len(stops_sequence)-1):
            passenger_in_bus = self._obtain_passengers_in_bus(stop_index=i, 
                                                            travel_time=total_travel_time+current_start_time,
                                                            bus_stops=stops_sequence,
                                                            stops_wait_time=stops_wait_time,
                                                            passenger_in_bus=passenger_in_bus,
                                                            stop_request_pairings=stop_request_pairings,
                                                            serviced_requests=serviced_requests,
                                                            request_capacities=request_capacities)
            current_location = stops_sequence[i]
            next_location = stops_sequence[i+1]
            next_index = i+1

            if passenger_in_bus + request_capacities[request_index] <= bus_capacity:
                deviation_result = self._insert_pickup_in_route(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=stops_sequence,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index)
                    
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                local_passengers_in_bus = copy.deepcopy(passenger_in_bus)
                local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    new_total_travel_time = total_travel_time + stops_wait_time[i] + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]
                    
                    local_passengers_in_bus = self._obtain_passengers_in_bus(stop_index=j,
                                                                            travel_time=full_travel_time+new_start_time,
                                                                            bus_stops=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            passenger_in_bus=local_passengers_in_bus,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            serviced_requests=local_serviced_requests,
                                                                            request_capacities=request_capacities)

                    total_passengers_in_bus = local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=new_start_time,
                                                                                total_travel_time=full_travel_time,
                                                                                stops_sequence=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                current_location=new_current_location,
                                                                                next_index=new_next_index,
                                                                                request_destination=request_destination,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=new_start_time,
                                                            stops_sequence=full_stop_sequence,
                                                            stops_wait_time=full_stops_wait_time,
                                                            stops_request_pair=full_stop_req_pair,
                                                            bus_location=full_stop_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            prev_passengers={},
                                                            consider_wait_time=consider_wait_times,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity)

                        total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair
                            
                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time += (stops_wait_time[i] + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def _place_request_online_exact(self, current_start_time, bus_capacity, bus_location, planned_stops, stops_wait_time, request_origin, 
                                    request_destination, requests_pickup_times, stop_request_pairings, passengers_in_bus, 
                                    prev_passengers, request_index, request_capacities, time_horizon, state_num, cost_func: str, current_bus_index: int, consider_wait_times=True, include_scaling=False,
                                    is_for_RV: bool = False, is_for_VV: bool = True):
        total_travel_time = 0
        min_cost = float("inf")
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        min_start_time = 0
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        rv_list: list[tuple[int, Routing_plan]] = [] 

        
        for i in range(len(planned_stops)-1):
            if i == 0 and bus_location != planned_stops[0]:
                local_passengers_in_bus = local_passengers_in_bus
                current_location = bus_location
            else:
                local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                                        travel_time=total_travel_time+current_start_time,
                                                                        bus_stops=planned_stops,
                                                                        stops_wait_time=stops_wait_time,
                                                                        passenger_in_bus=local_passengers_in_bus,
                                                                        stop_request_pairings=stop_request_pairings,
                                                                        serviced_requests=serviced_requests,
                                                                        request_capacities=request_capacities)
                current_location = planned_stops[i]

            next_location = planned_stops[i+1]
            next_index = i+1
            
            if local_passengers_in_bus  + request_capacities[request_index] <= bus_capacity:
                if i == 0 and bus_location != planned_stops[0]:
                    mismatched_flag = True
                else:
                    mismatched_flag = False
                
                deviation_result = self._insert_pickup_in_route_online(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=planned_stops,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index,
                                                                mismatched_flag=mismatched_flag)
                    
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                new_local_passengers_in_bus = copy.deepcopy(local_passengers_in_bus)
                new_local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    if i == 0 and bus_location != planned_stops[0]:
                        stop_time = 0
                    else:
                        stop_time = stops_wait_time[i]
                    new_total_travel_time = total_travel_time + stop_time + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    new_full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]

                    new_local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=j,
                                                                                travel_time=new_full_travel_time+current_start_time,
                                                                                bus_stops=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                passenger_in_bus=new_local_passengers_in_bus,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                serviced_requests=new_local_serviced_requests,
                                                                                request_capacities=request_capacities)

                    total_passengers_in_bus = new_local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=current_start_time,
                                                                            total_travel_time=new_full_travel_time,
                                                                            stops_sequence=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            current_location=new_current_location,
                                                                            next_index=new_next_index,
                                                                            request_destination=request_destination,
                                                                            requests_pickup_times=requests_pickup_times,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair
                        
                        new_route_cost = self._calculate_cost_of_route(stops_sequence=full_stop_sequence, 
                                                            stops_wait_time=full_stops_wait_time,
                                                            stops_request_pair=full_stop_req_pair,
                                                            request_capacities=request_capacities, 
                                                            passengers_in_bus=passengers_in_bus,
                                                            time_horizon=time_horizon,
                                                            state_num=state_num,
                                                            current_start_time = current_start_time,
                                                            prev_passengers = prev_passengers,
                                                            bus_location = bus_location,
                                                            cost_func = cost_func) 
                        
                        total_dev_cost =  new_route_cost

                        new_routing_plan = Routing_plan(bus_stops = full_stop_sequence,
                                                        stops_wait_times = full_stops_wait_time,
                                                        stops_request_pairing = Bus_stop_request_pairings(full_stop_req_pair),
                                                        assignment_cost = total_dev_cost,
                                                        start_time = current_start_time,
                                                        route = [],
                                                        route_edge_times = [],
                                                        route_stop_wait_time = []) #These nones will update automaticaly later!
                        
                        rv_list.append((current_bus_index, new_routing_plan)) 

                        if total_dev_cost < 0:
                            print("Request index = " + str(request_index))
                            print("New bus stops = " + str(full_stop_sequence))


                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair

                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            if i == 0 and bus_location != planned_stops[0]:
                current_wait_time = 0
            else:
                current_wait_time = stops_wait_time[i]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time, rv_list
    
    def static_insertion(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, stop_request_pairing, requests_pickup_times, request_capacities, 
                         request_origin, request_destination, request_index, consider_wait_time=True, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence)
        local_stops_wait_time = copy.deepcopy(stops_wait_time)
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing)

        deviation_result = self._place_request_offline_exact(current_start_time=current_start_time,
                                                             bus_capacity=bus_capacity,
                                                             stops_sequence=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_wait_times=consider_wait_time,
                                                             include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time
    
    def dynamic_insertion(self, current_start_time, current_stop_index, bus_capacity, passengers_in_bus, prev_passengers, bus_location,
                          stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, request_origin, request_destination, 
                          requests_pickup_times, request_index, time_horizon: int, state_num: int, cost_func: str, current_bus_index: int, is_full,
                          consider_wait_time=True, approximate=False, include_scaling=True, is_for_RV: bool = False, is_for_VV: bool = True):
        
        if is_full:
            local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
            local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
            local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])
        else:
            local_stops_sequence = copy.deepcopy(stops_sequence)
            local_stops_wait_time = copy.deepcopy(stops_wait_time)
            local_stop_request_pairing = copy.deepcopy(stop_request_pairing)

        deviation_result = self._place_request_online_exact(current_start_time=current_start_time, 
                                                            bus_capacity=bus_capacity,
                                                            bus_location=bus_location,
                                                             planned_stops=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             time_horizon=time_horizon,
                                                             state_num=state_num,
                                                             cost_func = cost_func,
                                                             consider_wait_times=consider_wait_time,
                                                             include_scaling=include_scaling,
                                                             current_bus_index = current_bus_index)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _, rv_list = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, rv_list

    #TODO rewrite and fix this method
    def unallocate(self, request_id: int, routing_plan: Routing_plan, bus_index: int,
                   requests_pickup_times: dict[int, int], requests_capacities: dict[int, int], time_horizon: int, state_num: int, 
                   current_start_time: list[int], current_stop_index: list[int], current_location: list[int], 
                   passengers_in_bus: list[int], prev_passengers: list[dict[int, list[int]]], stop_index) -> tuple[Routing_plan, SimRequest] | None:
        
        #TODO NEW ALLOCATION IS SLICED NOW!!!!!!!!!!
        new_allocation = copy.deepcopy(routing_plan) #SLICED

        #this is old code
        # new_allocation.bus_stops = new_allocation.bus_stops[stop_index:]
        # new_allocation.stops_wait_times = new_allocation.stops_wait_times[stop_index:]
        # #TODO shouldnt this be a the special tupe ?
        # new_allocation.stops_request_pairing.data = new_allocation.stops_request_pairing.data[stop_index:]#SLICED



        if request_id in prev_passengers[bus_index]:
            return None
        else:
            node_origin = None
            node_destination = None
            for pos, stop in enumerate(new_allocation.stops_request_pairing):
                if request_id in stop['pickup'] and pos != 0:
                    stop['pickup'].remove(request_id)
                    node_origin = new_allocation.bus_stops[pos]
                if request_id in stop['dropoff'] and pos != 0:
                    stop['dropoff'].remove(request_id)
                    node_destination = new_allocation.bus_stops[pos]

            pos = 1
            while pos < len(new_allocation.stops_request_pairing) - 1:
                stop = new_allocation.stops_request_pairing[pos]
                if (stop['pickup'] == [] and stop['dropoff'] == [-1]) or \
                        (stop['pickup'] == [-1] and stop['dropoff'] == []):
                    new_allocation.bus_stops.pop(pos)
                    new_allocation.stops_request_pairing.pop(pos)
                    new_allocation.stops_wait_times.pop(pos)
                else:
                    if stop['pickup'] == []:
                        stop['pickup'] = [-1]
                    if stop['dropoff'] == []:
                        stop['dropoff'] = [-1]
                    pos += 1

                new_allocation.stops_wait_times = self._update_stop_wait_times_unallocate(local_travel_time=current_start_time[bus_index],
                                                                                  stop_index=0,
                                                                                  stops_sequence=new_allocation.bus_stops,
                                                                                  stop_request_pairings=new_allocation.stops_request_pairing,
                                                                                  stops_wait_time=new_allocation.stops_wait_times,
                                                                                  requests_pickup_times=requests_pickup_times,
                                                                                  current_bus_location = current_location[bus_index])
                
                new_allocation.assignment_cost = self._calculate_cost_of_route(stops_sequence = new_allocation.bus_stops, 
                                                                               stops_wait_time = new_allocation.stops_wait_times, 
                                                                               stops_request_pair = new_allocation.stops_request_pairing.data, 
                                                                                request_capacities = requests_capacities, 
                                                                                passengers_in_bus = passengers_in_bus[bus_index], 
                                                                                time_horizon = time_horizon, 
                                                                                state_num = state_num, 
                                                                                current_start_time = current_start_time[bus_index], 
                                                                                prev_passengers = prev_passengers[bus_index], 
                                                                                bus_location = current_location[bus_index], 
                                                                                cost_func = 'PTT') #TODO synchornise goof cost function!
            #TODO think what happens when this is None down the callstack!
            if (node_origin is None) or (node_destination is None):
                return None
            else:
                return new_allocation, SimRequest(node_origin, node_destination, request_id)
            

    def _update_stop_wait_times_unallocate(self, local_travel_time, stop_index, stops_sequence, 
                                        stop_request_pairings, stops_wait_time, requests_pickup_times,
                                        current_bus_location: int,
                            default_stop_wait_time=60):
    
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(stop_index, len(new_stops_wait_time)-1):
            if current_bus_location != stops_sequence[0] and i == 0:
                next_location = stops_sequence[i+1]
                current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_bus_location, next_location)
                local_travel_time += current_edge_cost
            else:
                current_request_index_dict = stop_request_pairings[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                for list_index, current_request_index in enumerate(pickup_requests_list):
                    if current_request_index == -1:
                        continue
                    else:
                        if list_index == 0:
                            new_stops_wait_time[i] = default_stop_wait_time
                        current_request_pickup_time = requests_pickup_times[current_request_index]
                        new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
                current_location = stops_sequence[i]
                wait_time = new_stops_wait_time[i]
                next_location = stops_sequence[i+1]
                current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
                local_travel_time += (wait_time + current_edge_cost)
        
        return new_stops_wait_time
