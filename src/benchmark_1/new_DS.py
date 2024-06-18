import copy
import textwrap
from dataclasses import dataclass, field
import pandas as pd
from typing import Any, Counter, Generator, Iterator, NamedTuple
from datetime import datetime
import benchmark_1.config as config

from State import State

@dataclass
class SimRequest:
    '''info to describe request for simulator'''
    origin: int
    destination: int
    index: int
    creation_time: datetime | None = None
    pickup_time: int | None = None
    capacity: int | None = None
    pickup_times_timestamp: datetime | None = None

    def __repr__(self) -> str:
        '''string representation of the object'''
        return f'req={self.index}'
        #if you need more expressive printing
        # return f'SimRequest(origin={self.origin}, destination={self.destination}, index={self.index})'

    def __hash__(self):
        return hash((self.origin, self.destination, self.index))

class SimRequestChain(NamedTuple):
    '''imuttable, list of requests that are chained together'''
    chain: list[SimRequest]

    def reached_end(self) -> bool:
        '''check if request chain is empty'''
        return self.chain == []
    
    def from_depth(self, depth: int) -> SimRequest:
        '''create a specific request chain based on the tree depth'''
        return self.chain[depth]

    def __getitem__(self, index) -> 'SimRequestChain | SimRequest':
        if isinstance(index, slice):
            return SimRequestChain(self.chain[index])
        else:
            return self.chain[index]
    
    def __hash__(self):
        return hash(tuple(self.chain))
    
    def append_left(self, new_request: SimRequest) -> None:
        '''inplace!'''
        return self.chain.insert(0, new_request)

    def __repr__(self) -> str:
        return 'SimReqChain' + str(self.chain)


class Route:
    '''action our algorithm can take; planned route for a single bus'''

    def __init__(self, stops_sequence_future: list[int], 
                 stops_wait_times_future: list[int], 
                 stop_req_pairs_future: list[dict[str, list[int]]], 
                 assignment_cost_future: int | None = None) -> None:
        self.stops_sequence_future = stops_sequence_future
        self.stops_wait_times_future = stops_wait_times_future
        self.stop_req_pairs_future = stop_req_pairs_future
        self.assignment_cost_future = assignment_cost_future 

    def __hash__(self) -> int:
        '''hash the object'''
        return hash((tuple(self.stops_sequence_future),
                    tuple(self.stops_wait_times_future)))

    def get_unpicked_req_ids(self) -> list[int]:
        '''inplace, generator to iterate over all requests that have not been
        picked up so far in the route'''
        combine = [stop['pickup'] + stop['dropoff']
                   for stop in self.stop_req_pairs_future]
        flatten = [item for sublist in combine for item in sublist]
        counts = Counter(flatten)
        repeated_values = [request for request,
                           count in counts.items() if count == 2]
        return repeated_values

    def allocate(self, request: SimRequest, state_object: State, bus_index: int) -> None:
        '''INPLACE, greedy, allocate request to a bus at a greedily selected the location 
        where utility will be maximized'''
        req = Request_Insertion_Procedure_baseline_1(map_graph=config.MAP_GRAPH)

        local_route = Route(copy.deepcopy(self.stops_sequence_future),
                            copy.deepcopy(self.stops_wait_times_future), 
                            copy.deepcopy(self.stop_req_pairs_future), 
                            0) #TODO how to deal with the yet unassigned route cost?

        out = req.place_request_online_exact(bus_index, 
                                             local_route, 
                                             request, 
                                             state_object, 
                                             is_for_VV = True)
 
        min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time = out

        self.stops_sequence_future = min_stop_sequence
        self.stops_wait_times_future = min_stop_wait_times
        self.stop_req_pairs_future = min_stop_request_pairings
        self.assignment_cost_future = min_cost

    def unallocate(self, request_id: int, route: 'Route', state_object: State, bus_index: int) -> SimRequest | None:
        '''INPLACE,  unallocate request from the selected bus route;
        first remove requests from the step_req_pairs_future list and then remove the  empty 
        stops from stops sequence, stops wait times and stop_req_pairs_future lists;
        build requests based on how you retrieve'''

        if request_id in state_object.correct_state.prev_passengers[bus_index]:
            return None
        else:
            node_origin = None
            node_destination = None
            for pos, stop in enumerate(route.stop_req_pairs_future):
                if request_id in stop['pickup'] and pos != 0:
                    stop['pickup'].remove(request_id)
                    node_origin = route.stops_sequence_future[pos]
                if request_id in stop['dropoff'] and pos != 0:
                    stop['dropoff'].remove(request_id)
                    node_destination = route.stops_sequence_future[pos]

            pos = 1 #TODO this doesnt make sense change this in the future
            while pos < len(route.stop_req_pairs_future) - 1:
                stop = route.stop_req_pairs_future[pos]
                if (stop['pickup'] == [] and stop['dropoff'] == [-1]) or \
                        (stop['pickup'] == [-1] and stop['dropoff'] == []):
                    route.stops_sequence_future.pop(pos)
                    route.stop_req_pairs_future.pop(pos)
                    route.stops_wait_times_future.pop(pos)
                else:
                    if stop['pickup'] == []:
                        stop['pickup'] = [-1]
                    if stop['dropoff'] == []:
                        stop['dropoff'] = [-1]
                    pos += 1

                req = Request_Insertion_Procedure_baseline_1(map_graph=config.MAP_GRAPH)
                route.stops_wait_times_future = req._update_stop_wait_times_unallocate(local_travel_time=state_object.correct_state.current_start_time[bus_index],
                                                                                    stop_index=0,
                                                                                    stops_sequence=route.stops_sequence_future,
                                                                                    stop_request_pairings=route.stop_req_pairs_future,
                                                                                    stops_wait_time=route.stops_wait_times_future,
                                                                                    requests_pickup_times=state_object.requests_pickup_times,
                                                                                    current_bus_location = state_object.correct_state.new_bus_location[bus_index])
            if (node_origin is None) or (node_destination is None):
                return None
            else:
                return SimRequest(node_origin, node_destination, request_id)

    def __lt__(self, other: 'Route') -> bool:
        '''compare two Route instances'''
        return self.assignment_cost_future < other.assignment_cost_future
    
    def __repr__(self) -> str:
        width = 40  # Maximum width of the string representation
        
        stops_sequence_str = textwrap.shorten(str(self.stops_sequence_future), width=width, placeholder="...")
        wait_times_str = textwrap.shorten(str(self.stops_wait_times_future), width=width, placeholder="...")
        req_pairs_str = textwrap.shorten(str(self.stop_req_pairs_future), width=width, placeholder="...")
        
        return (f"Route(stops_sequence={stops_sequence_str}, "
                f"wait_times={wait_times_str}, "
                f"req_pairs={req_pairs_str}, "
                f"cost={self.assignment_cost_future})")

class Buses(NamedTuple):
    '''list of all routes assigned to buses'''
    routes: list[Route]

    def compute_utility(self) -> int:
        '''sum of all utilities of the routes'''
        return sum(route.assignment_cost_future for route in self.routes)

    def __getitem__(self, index: int) -> Route:
        '''get route at index'''
        return self.routes[index]

    def __setitem__(self, index: int, value: Route) -> None:
        self.routes[index] = value

    def __hash__(self) -> int:
        '''hash the object'''
        return hash(tuple(self.routes))

    @classmethod
    def from_state(self, state_object: State, current_stop_index: list[int]) -> 'Buses':
        '''build Buses from state_object'''
        '''create Buses object from state object'''
        buses = []
        for bus_index in range(config.NUM_BUSSES):
            buses.append(Route(state_object.bus_stops[bus_index][current_stop_index[bus_index]:],
                               state_object.stops_wait_times[bus_index][current_stop_index[bus_index]:],
                               state_object.stops_request_pairing[bus_index][current_stop_index[bus_index]:],
                               state_object.assignment_cost[bus_index])) #TODO this bus path doesn't make to much sense
        return Buses(buses)

    def unpack_for_simulator(self) -> tuple[list[int], list[list[int]], list[list[int]], list[list[dict[str, list[int]]]]]:
        '''convert back to data compatible with the DS in the simulator'''
        return [route.assignment_cost_future for route in self.routes], \
               [route.stops_sequence_future for route in self.routes], \
               [route.stops_wait_times_future for route in self.routes], \
               [route.stop_req_pairs_future for route in self.routes]
    

    
from Map_graph import Map_graph

class Request_Insertion_Procedure_baseline_1:
    def __init__(self, map_graph: Map_graph) -> None:
        self.map_graph = map_graph
    
    #TODO add method wher we look at the state_object from some perspective
    def _calculate_cost_of_route(self, route: Route, state_object: State, bus_index: int, cost_func: str = 'PTT') -> int:
        '''compute the costs outlined in the papers based on the single Route of the bus'''
        #TODO CAPACITIES FOR THE HISTORIC REQUESTS KEEP IT SOMEHWERE!
        route_cost = 0
        travel_time = [self.map_graph.obtain_shortest_paths_time(route.stops_sequence_future[stp],
                                                           route.stops_sequence_future[stp+1])
                                                           for stp in range(len(route.stops_sequence_future)-1)]
        time_in_bus = [travel + wait for travel, wait in zip(travel_time, route.stops_wait_times_future)]
        num_passengers_stops = [state_object.correct_state.passengers_in_bus[bus_index]] #TODO should this be correct state or regular state?
        #reuqests capacities historic must be made for the historical dataset
        for stop in route.stop_req_pairs_future:
            pick = sum([state_object.request_capacities[request_id] for request_id in stop["pickup"] if request_id != -1]) #TODO how to get requests capacities from the major requests
            drop = sum([-state_object.request_capacities[request_id] for request_id in stop["dropoff"] if request_id != -1])
            net_change = pick+drop
            total_change = num_passengers_stops[-1] + net_change
            if total_change < 0:
                print('DAMAGED NODE')
                print(state_object.correct_state.passengers_in_bus)
                print(route.stop_req_pairs_future)
            assert total_change >= 0, f'Passenger Number must be positive, failed for {route.stop_req_pairs_future}'
            num_passengers_stops.append(total_change) #TODO add to the previours value
        match cost_func:
            case 'PTT':
                route_cost = sum([time*passengers for time, passengers in zip(time_in_bus, num_passengers_stops)])
            case 'future requests budget':
                num_passengers_stops_step_mask = [1 if passenger else 0 for passenger in num_passengers_stops] #TODO add to the previours value
                sumation = sum([time*passengers for time, passengers in 
                                zip(time_in_bus, num_passengers_stops_step_mask)])
                route_cost =  state_object.time_horizon - state_object.state_num - sumation #QUESTION DANIEL should this state_num be also changed to .correct_state
            case _:
                raise ValueError('You passed an undefined cost function type!')
        assert route_cost >= 0, 'Route cost cannot be negative!'
        return route_cost
    
    def _place_request_inside_stop(self, local_stop_request_pairings, stop_index, request_index, label):
        try: 
           local_stop_request_pairings[stop_index][label][0]
        except IndexError:
            a = 10 
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
    
    def _update_stop_wait_times(self, local_travel_time, stop_index, stops_sequence,    stop_request_pairings, stops_wait_time, requests_pickup_times,
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
    
    def place_request_online_exact(self, bus_index: int, route: Route, request: SimRequest, state_object: State, consider_wait_times: bool=False, include_scaling: bool=False, 
                                    is_for_RV: bool= False,
                                    is_for_VV: bool= False) -> Generator[tuple[int, Route], None, None] | tuple[int, list[int], list[int], dict[str, list[int]]]:
        total_travel_time = 0
        min_cost = float("inf")
        min_stop_sequence = []
        min_stop_wait_times = [] 
        min_stop_request_pairings = []
        min_start_time = 0
        RV_edge_list = []
        serviced_requests: dict[int, list[int]] = copy.deepcopy(state_object.correct_state.prev_passengers[bus_index])
        local_passengers_in_bus: int = copy.deepcopy(state_object.correct_state.passengers_in_bus[bus_index])
        
        original_route_cost = self._calculate_cost_of_route(route, state_object, bus_index)
        
        for i in range(len(route.stops_sequence_future)-1):
            if i == 0 and state_object.correct_state.new_bus_location[bus_index] != route.stops_sequence_future[0]:
                local_passengers_in_bus = local_passengers_in_bus
                current_location = state_object.correct_state.new_bus_location[bus_index]
            else:
                local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                                        travel_time=total_travel_time+state_object.correct_state.current_start_time[bus_index],
                                                                        bus_stops=route.stops_sequence_future,
                                                                        stops_wait_time=route.stops_wait_times_future,
                                                                        passenger_in_bus=local_passengers_in_bus,
                                                                        stop_request_pairings=route.stop_req_pairs_future,
                                                                        serviced_requests=serviced_requests,
                                                                        request_capacities=state_object.request_capacities)
                current_location = route.stops_sequence_future[i]

            next_location = route.stops_sequence_future[i+1]
            next_index = i+1
            
            if local_passengers_in_bus  + state_object.request_capacities[request.index] <= state_object.bus_capacities[bus_index]:
                if i == 0 and state_object.correct_state.new_bus_location[bus_index] != route.stops_sequence_future[0]:
                    mismatched_flag = True
                else:
                    mismatched_flag = False
                
                deviation_result = self._insert_pickup_in_route_online(current_start_time=state_object.correct_state.current_start_time[bus_index],
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=route.stops_sequence_future,
                                                                stops_wait_time=route.stops_wait_times_future, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request.origin, 
                                                                requests_pickup_times=state_object.requests_pickup_times,
                                                                stop_request_pairings=route.stop_req_pairs_future,
                                                                request_index=request.index,
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
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request.origin)
                    if i == 0 and state_object.correct_state.new_bus_location[bus_index] != route.stops_sequence_future[0]:
                        stop_time = 0
                    else:
                        stop_time = route.stops_wait_times_future[i]
                    new_total_travel_time = total_travel_time + stop_time + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    new_full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]

                    new_local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=j,
                                                                                travel_time=new_full_travel_time+state_object.correct_state.current_start_time[bus_index],
                                                                                bus_stops=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                passenger_in_bus=new_local_passengers_in_bus,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                serviced_requests=new_local_serviced_requests,
                                                                                request_capacities=state_object.request_capacities)

                    total_passengers_in_bus = new_local_passengers_in_bus
                    if total_passengers_in_bus > state_object.bus_capacities[bus_index]:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=state_object.correct_state.current_start_time[bus_index],
                                                                            total_travel_time=new_full_travel_time,
                                                                            stops_sequence=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            current_location=new_current_location,
                                                                            next_index=new_next_index,
                                                                            request_destination=request.destination,
                                                                            requests_pickup_times=state_object.requests_pickup_times,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            request_index=request.index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        final_route = Route(full_stop_sequence, full_stops_wait_time, full_stop_req_pair)
                        new_route_cost = self._calculate_cost_of_route(final_route, state_object, bus_index) #TODO what should i pass here?

                        #TODO what cost function should i be using?
                        total_dev_cost =  new_route_cost # - original_route_cost
                        
                        if is_for_RV:
                            RV_edge_list.append((bus_index, Route(full_stop_sequence, full_stops_wait_time, full_stop_req_pair, new_route_cost))) 

                        # TODO enable this if cost can be negative!

                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair
                    
                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            if i == 0 and state_object.correct_state.new_bus_location[bus_index] != route.stops_sequence_future[0]:
                current_wait_time = 0
            else:
                current_wait_time = route.stops_wait_times_future[i]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

        if is_for_RV:
            return RV_edge_list

        if is_for_VV:
            return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def dynamic_insertion(self, current_start_time, current_stop_index, bus_capacity, passengers_in_bus, prev_passengers, bus_location,
                          stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, request_origin, request_destination, 
                          requests_pickup_times, request_index, consider_wait_time=True, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
        local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])

        deviation_result = self.place_request_online_exact(current_start_time=current_start_time, 
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
                                                            consider_wait_times=consider_wait_time,
                                                            include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _ = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair
