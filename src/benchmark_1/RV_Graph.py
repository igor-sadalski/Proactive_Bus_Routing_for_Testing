from collections import Counter
import copy
from typing import Generator
from heapq import heappop, heappush

from Data_structures import Routing_plan
import benchmark_1.config as config
from benchmark_1.new_DS import Route, SimRequest, Buses, Request_Insertion_Procedure_baseline_1
from State import State

from typing import Callable

import pandas as pd

class RVGraph:
    '''retriev values with highest utility'''

    def __init__(self, buses_paths: list[Routing_plan], request: SimRequest, 
                 assignment_policy: Callable[[list[Routing_plan]], tuple[int, int, Routing_plan, list[int]]]) -> None:
        
        assert self._is_healthy(buses_paths), 'more than two idential requests per one stop'
        self.request = request
        self.E_RV: list[list[Routing_plan]] = [[] for _ in range(config.NUM_BUSSES)] #full sequence! not sliced!

        all_possible_rv_paths: list[tuple[int, Routing_plan]]
        *_, all_possible_rv_paths = assignment_policy(plans=buses_paths, 
                                                      current_request_index=request.index, 
                                                      current_request_row={"Origin Node": request.origin,
                                                                           "Destination Node": request.destination}) #TODO what about request time when it comes to the system
        
        for (bus_index, possible_routing_plan) in all_possible_rv_paths:
            self.E_RV[bus_index].append(possible_routing_plan)


            
    def edge_iterator(self) -> Generator[tuple[int, Routing_plan], None, None]:
        '''generator to iterate over all edges in the RV graph'''
        if self.E_RV:
            for vehicle_id, edges in enumerate(self.E_RV):
                for route in edges:
                    yield vehicle_id, route
        else:
            raise ValueError('RVGraph is empty')
        
    def get_min_PTT_edge(self) -> tuple[int, Routing_plan]:
        '''get highest utility edge without swapping requests;
        scan over all buses checking what minumum value they have;
        th'''
        if any(self.E_RV[bus_num] == [] for bus_num in range(3)):
            print('empty entry in RV')
        #TODO ASK daniel I keep getting empyt RV graphs!
        mins = [(bus_num, min(self.E_RV[bus_num])) for bus_num in range(config.NUM_BUSSES) if self.E_RV[bus_num]] #TODO why this can be zero sometimes...
        if not mins:
            print('empty RV') #TODO how could this be possible?
        min_bus, min_route = min(mins, key=lambda x: (x[1].assignment_cost 
                                                if (x[1].assignment_cost is not None) else 0))
        return min_bus, min_route
    
        #TODO this should be the method of routing plan to check if it is not corrupted
    def _is_healthy(self, buses_paths: list[Routing_plan]) -> bool:
        all_values = tuple([item for routing_plan in buses_paths 
                                 for stop in routing_plan.stops_request_pairing.data 
                                 for value in stop.values() 
                                 for item in value
                                 if item != -1])
        counts = Counter(all_values).values()
        return all([count <= 2 for count in counts])    

    #TODO add better repre string format!
    def  __repr__(self) -> str:
        return self.E_RV
    #TODO add meaningful representation for this 