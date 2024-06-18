from collections import Counter
from copy import deepcopy
from heapq import heappush, heappop

from benchmark_1.RV_Graph import RVGraph
from benchmark_1.VV_Graph import VVGraph
import benchmark_1.config as config
from benchmark_1.new_DS import Buses, SimRequest
from Data_structures import Routing_plan
from typing import Callable

class Actions:
    '''implementation of algo 1 used to generate promising actions during the 
    expansion phase of building the MCTS tree'''
    def __init__(self, 
                 routing_plans: list[Routing_plan],
                 allocate_method_vv: Callable[[SimRequest, list[Routing_plan]], tuple[int, Routing_plan, list[int]]], 
                 unallocate_method_vv: Callable[[int, Routing_plan, int], tuple[Routing_plan, SimRequest]],
                 current_stop_index: list[int], 
                 greedy_assignment_rv: Callable[[list[Routing_plan]], tuple[int, int, Routing_plan, list[int]]],
                 request: SimRequest) -> None:
        
        # if not self._is_healthy(routing_plans):
        #     print('wrong')
        assert self._is_healthy(routing_plans), 'more than two idential requests per one stop'

        self.rv_graph = RVGraph(routing_plans, request, greedy_assignment_rv)
        self.vv_graph = VVGraph(routing_plans, allocate_method_vv,
                                unallocate_method_vv, current_stop_index)
        self.request = request
        self.theta: list[Routing_plan] = routing_plans
        self.heap: list[tuple[int, list[Routing_plan]]] = []
        self.promising = self._algo1()

    #TODO after this is done you can remove from memory vv and rv graphs
    def _algo1(self) -> list[list[Routing_plan]]:
        '''notation taken from paper; iterate over rv edges for each edge compute its path
        utility and push it onto the heap, delete this rv vertex from the graph; then iterate
        over vv edges and sequentialy attach the use the larges vv edges to autment the initial path
        after each augmentation push the new path onto the heap and delete the verticies that were 
        used from the grpah'''
        for vehicle_id, er_ij_path in self.rv_graph.edge_iterator():
            updated_theta_ij = deepcopy(self.theta)
            updated_theta_ij[vehicle_id] = er_ij_path
            u_x = sum(routing_plan.assignment_cost for routing_plan in updated_theta_ij)
            assert u_x >= 0, 'ptt cost cannot be negative'
            heappush(self.heap, (u_x, updated_theta_ij)) #TODO what happens when utility is negative
            vv_copy = deepcopy(self.vv_graph)
            vv_copy.delete_vertex(vehicle_id)
            while vv_copy.is_non_empty():
                m, n, vv_edge = vv_copy.arg_min()
                u_x += vv_edge.swap_utility #TODO can this be negative?
                updated_theta_ij[m] = vv_edge.bus_m_route
                updated_theta_ij[n] = vv_edge.bus_n_route
                heappush(self.heap, (u_x, updated_theta_ij)) #TODO two values with the same utility will be heappoped at random
                vv_copy.delete_vertex(m, n)
        k_smallest = [heappop(self.heap)[1] for _ in range(min(config.K_MAX, len(self.heap)))]
        return k_smallest
    
    #TODO this should be the method of routing plan to check if it is not corrupted
    def _is_healthy(self, buses_paths: list[Routing_plan]) -> bool:
        all_values = tuple([item for routing_plan in buses_paths 
                                    for stop in routing_plan.stops_request_pairing.data 
                                    for value in stop.values() 
                                    for item in value
                                    if item != -1])
        counts = Counter(all_values).values()
        return all([count <= 2 for count in counts])    