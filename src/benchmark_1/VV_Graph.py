from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from heapq import heapify
from typing import Callable, Generator

from Data_structures import Routing_plan
# from State import State, CorrectState
from benchmark_1.new_DS import Buses, Route, Request_Insertion_Procedure_baseline_1, SimRequest
import benchmark_1.config as config

import pandas as pd

from benchmark_1.utilities import log_runtime_and_memory


@dataclass
class VVEdge:
    '''edge of vvgraph defined in this way to support
    operations required by the algorithm 1'''
    bus_m_route: Routing_plan
    bus_n_route: Routing_plan
    swap_utility: int


class AdjecenyList:
    '''representation of our graph'''

    def __init__(self) -> None:
        self.ll: dict[int, dict[int, VVEdge]] = {bus_index: {} for bus_index in range(config.NUM_BUSSES)}

    def insert_edge_min(self, m: int, n: int, edge: VVEdge) -> None:
        '''insert VVEdge into VVGraph'''
        self.ll[m][n] = edge

    def delete_vertex(self, *vehicles: int) -> None:
        '''remove dict row where each vehicle in vehicles is the key then iterate over 
        all other keys and remove the inner key which is equal to each vehicle in vehicles;
        support deletion of multiple verticies in the graph'''
        for vehicle in vehicles:
            self.ll.pop(vehicle, None)
            for vehicle_key in self.ll:  # type: ignore
                self.ll[vehicle_key].pop(vehicle, None)

    def min_global_edge(self) -> tuple[int, int, VVEdge]:
        '''return edge with min utility from all edges in the graph!'''
        return min(((m, n, value) for m, sub_dict in self.ll.items()
                                  for n, value in sub_dict.items()),
                                  key=lambda x: x[2].swap_utility)


class VVGraph:
    '''represented the graph as an adjeceny list with
    dictionaries, support some operations required by algo 1'''

    def __init__(self, buses_paths: list[Routing_plan], 
                 allocate_method: Callable[[SimRequest, list[Routing_plan]], tuple[int, Routing_plan, list[int]]],
                 unallocate_method: Callable[[int, Routing_plan, int], tuple[Routing_plan, SimRequest] | tuple[None, None]],
                 stop_index: list[int]) -> None:
        '''iterate over all possible bus pair for each pair select the best request to swap'''
        self.E_VV = AdjecenyList()
        bus_gen = ((m, n) for m in range(config.NUM_BUSSES)
                          for n in range(config.NUM_BUSSES)
                          if m != n)
        for m, n in bus_gen:
            bus_m = buses_paths[m] #partial
            bus_n = buses_paths[n] #partial
            original_cost = bus_m.assignment_cost + bus_n.assignment_cost
            cur_min_edge = VVEdge(bus_m, bus_n, float('inf'))
            stop_index_at_m = stop_index[m]
            for unpicked_req_id in bus_m.get_unpicked_req_ids(stop_index_at_m):
                out = unallocate_method(request_id = unpicked_req_id, 
                                        routing_plan = deepcopy(bus_m), 
                                        bus_index =  m,
                                        stop_index = stop_index_at_m) #TODO what values do the unallocate work on ? SLICED or unliced?!!!!!
                new_bus_m, unallocated_request = out or (None, None)
                if unallocated_request:
                    #TODO this bus index gets automaticaly rested to 0!
                    _, new_bus_n, _ = allocate_method(request_index=unpicked_req_id, 
                                                      request_row={"Origin Node": unallocated_request.origin,
                                                                   "Destination Node": unallocated_request.destination},  
                                                      plans = [deepcopy(bus_n), deepcopy(bus_n), deepcopy(bus_n)],
                                                      bus_index = n,
                                                      current_bus_index = n) #TODO will this break it? QUESTION DANIEL
                    new_route_cost = new_bus_m.assignment_cost + new_bus_n.assignment_cost
                    final_utlity = new_route_cost - original_cost
                    new_edge = VVEdge(new_bus_m, new_bus_n, final_utlity)
                    cur_min_edge = min(new_edge, cur_min_edge, 
                                        key=lambda x: x.swap_utility)
            vv_edge = cur_min_edge if cur_min_edge.swap_utility != float('inf') else None
            if vv_edge:
                self.E_VV.insert_edge_min(m, n, vv_edge)

    def is_non_empty(self) -> bool:
        '''check if there are at least two edges in the VVGraph'''
        return bool(self.E_VV.ll) and any(self.E_VV.ll.values())

    def arg_min(self) -> tuple[int, int, VVEdge]:
        '''find edge with minimum utility'''
        return self.E_VV.min_global_edge()

    def delete_vertex(self, *vehicles: int) -> None:
        '''remove any number of passed vertex from VV graph'''
        self.E_VV.delete_vertex(*vehicles)
