from copy import deepcopy
from datetime import datetime
import math as m
import os
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, deque

from benchmark_1.new_DS import SimRequest, SimRequestChain
from benchmark_1.RV_Graph import RVGraph
from benchmark_1.algo1 import Actions
from benchmark_1.generative_model import GenerativeModel
import benchmark_1.config as config

from Data_structures import Routing_plan
from typing import Callable, Optional

from benchmark_1.utilities import log_runtime_and_memory

verbose = False

@dataclass
class MCNode:
    request: SimRequest
    buses_paths: list[Routing_plan]
    parent: 'Optional[MCNode]' = None #TODO this cost is note getting updated during rolout!
    children: list['MCNode'] = field(default_factory=list)
    visits: int = 1
    reward: list[float] = field(default_factory=list)

    def compute_ucb(self) -> float:
        '''update visits, avg_value and avg in this order'''
        parent = self.parent.visits if self.parent else 1
        current = self.visits
        freq_term = (m.sqrt(m.log(parent) / current))
        avg = sum(self.reward) / len(self.reward)
        return avg + config.MCTS_TUNING_PARAM * freq_term

    def update_visits(self) -> None:
        self.visits += 1

    def select_best_child(self) -> 'MCNode':
        if self.children:
            return max(self.children, key=lambda x: x.compute_ucb()) #TODO we are selecting incorrrect value!
        else:
            raise ValueError('There are no children in this list')

    def append_child(self, node: 'MCNode') -> None:
        self.children.append(node)
    
    def __repr__(self) -> str:
        return f'''req={self.request.index}, reward={self.reward}, visits={self.visits}, parent={self.parent.request.index if self.parent else None}, children={[child.request.index for child in self.children] if self.children else None}, bus_path={[path.stops_request_pairing for path in self.buses_paths]}'''

class MCTree:
    def __init__(self,
                 initial_routing_plans: list[Routing_plan],
                 initial_request: SimRequest,
                 allocate_method_vv: Callable[[SimRequest, list[Routing_plan]], tuple[int, Routing_plan, list[int]]],
                 unallocate_method_vv: Callable[[int, Routing_plan, int], tuple[Routing_plan, SimRequest]],
                 current_stop_index: list[int],
                 greedy_assignment_rv: Callable[[list[Routing_plan]], tuple[int, int, Routing_plan, list[int]]]) -> None:

        self.allocate_method_vv = allocate_method_vv
        self.unallocate_method_vv = unallocate_method_vv
        self.current_stop_index = current_stop_index
        self.greedy_assignment_rv = greedy_assignment_rv
        self.root: MCNode = MCNode(initial_request, initial_routing_plans)

    def select(self, node: MCNode | None = None, depth=+0) -> tuple[int, MCNode]:
        '''starting at root recursively select child with highest
        UCB value; at leaf return this child'''
        node = node if node else self.root
        if verbose:  print('select: ', node.request.index, hash(tuple(node.buses_paths)))
        node.visits += 1
        if node.children and depth < config.MCTS_DEPTH:
            return self.select(node.select_best_child(), depth+1)
        else:
            return depth, node

    def expand(self, current_node: MCNode, future_generated_request: SimRequest) -> None:
        best_actions = Actions(current_node.buses_paths,
                               self.allocate_method_vv,
                               self.unallocate_method_vv,
                               self.current_stop_index,
                               self.greedy_assignment_rv,
                               current_node.request) #we compute actions based o the requests in the given node
        for bus_path_action in best_actions.promising:
            new_node = MCNode(request = future_generated_request,
                              buses_paths = bus_path_action,
                              parent = current_node,
                              visits = 1,
                              reward = [])
            current_node.append_child(new_node)

    def rollout(self, buses_paths: list[Routing_plan], requests_chain: SimRequestChain) -> int:
        '''recurse down while we have requests (this should be equal to tree depth); when you
        run of out requests compute the utility of of the BusesPaths; updated rolled out
        nodes with this values'''
        if requests_chain.reached_end():
            total = sum(routing_plan.assignment_cost for routing_plan in buses_paths)
            if verbose:  print('rollout end cost', total) 
            return total
        else:
            total_for_checks = sum(routing_plan.assignment_cost for routing_plan in buses_paths)
            if verbose:  print('rollout', requests_chain.chain[0].index, hash(tuple(buses_paths)), total_for_checks)
            #TODO how to deal when rV graph cannot return anything?
            bus_index, best_edge = RVGraph(buses_paths,
                                           requests_chain.chain[0],
                                           self.greedy_assignment_rv).get_min_PTT_edge() #TODO can creating an RV graph be impossible?
            new_theta = deepcopy(buses_paths)
            new_theta[bus_index] = best_edge
            final_cost = self.rollout(new_theta,
                                      SimRequestChain(requests_chain.chain[1:]))
            return final_cost

    def backpropagate(self, value: float, node: MCNode) -> None:
        '''recurse up the tree and update average of each
        tree node in the path; no need to update root as we will always select
        it anyways; this will be initialy called on a already rolled out node
        to backprop its value through the whole stack'''
        if node:
            if verbose:  print('backprop', node.request.index, hash(tuple(node.buses_paths)))
            node.reward.append(value) #TODO check if i should start at this node or somewhere in between?
            if node.parent:
                self.backpropagate(value, node.parent)

    def __repr__(self) -> str:
        '''if verbose:  print level wise nodes with their index and ucb selection value'''
        level = deque(self.root.children)
        empty = '(-, -, -)'
        out = '->0lvl | ' + '      ' * int(2**(config.MCTS_DEPTH - 1))
        #TODO shouldi have some itinital reard in the root before i start exporing the tree
        out += f'({self.root.request.index}, {self.root.reward}, {int(self.root.visits):d}), {hash(tuple(self.root.buses_paths))}' + '\n'
        lvl_ind = 1
        while any(val != empty for val in level):
            out += f'->{lvl_ind}lvl | ' 
            next_level = deque([])
            while level:
                child = level.popleft()
                out += '      ' * int(2**(config.MCTS_DEPTH - lvl_ind - 1))
                if child != empty:
                    out += f'({child.request.index}, {child.reward}, {int(child.visits):d}, {hash(tuple(child.buses_paths))}) '
                    if child.children:
                        next_level.extend(child.children)
                    next_level.extend([empty for _ in range(config.K_MAX - len(child.children))])
                else:
                    out += empty
                    next_level.extend([empty for _ in range(config.K_MAX)])
                out += '      ' * int(2**(config.MCTS_DEPTH - lvl_ind - 1)-1)
            out += '\n'
            lvl_ind += 1 
            level = next_level
        return out


class MCForest:
    '''compute predicted utility of each actions we can take from our present state;
    in parallel evaluate trees and update predicted action utilites; in the end take the action
    with highest present and future utlity'''

    def __init__(self,
                 initial_routing_plans: list[Routing_plan],
                 initial_request: SimRequest,
                 allocate_method_vv: Callable[[SimRequest, list[Routing_plan]], tuple[int, Routing_plan, list[int]]],
                 unallocate_method_vv: Callable[[int, Routing_plan, int], tuple[Routing_plan, SimRequest]],
                 current_stop_index: list[int],
                 greedy_assignment_rv: Callable[[list[Routing_plan]], tuple[int, int, Routing_plan, list[int]]],
                 generative_model: GenerativeModel) -> None:

        self.initial_request = initial_request
        self.initial_routing_plans = initial_routing_plans
        self.allocate_method_vv = allocate_method_vv
        self.unallocate_method_vv = unallocate_method_vv
        self.current_stop_index = current_stop_index
        self.greedy_assignment_rv = greedy_assignment_rv
        self.generative_model: GenerativeModel = generative_model

    def _evaluate_tree(self, time_out: int = 240) -> MCTree:
        '''perform multiple iterations of selection -> expansion -> rollout -> backprop;
        move the dummy pointer from root to selected node; expand that node; '''
        if verbose:  print('--------------------------------------------------------------------')
        requests = self.generative_model.sample()
        if verbose:  print('chain', requests)
        root = dummy = MCTree(initial_routing_plans = self.initial_routing_plans,
                              initial_request = self.initial_request,
                              allocate_method_vv = self.allocate_method_vv,
                              unallocate_method_vv = self.unallocate_method_vv,
                              current_stop_index = self.current_stop_index,
                              greedy_assignment_rv = self.greedy_assignment_rv)
        if verbose:  print('initial tree\n', root)
        time_start = datetime.now()

        for _ in range(config.SINGLE_MCTREE_ITERATIONS):
            current_time = datetime.now()
            time_difference = (current_time - time_start).total_seconds()
            if time_difference <= time_out:
                dummy = root
                if verbose:  print('START NEXT S-E-R-B ITERATION')
                if verbose:  print('before select\n', root)
                depth, selected_node = dummy.select()
                if depth == config.MCTS_DEPTH:
                    one_node_up = selected_node.parent 
                    if one_node_up:
                        assert len(selected_node.reward) == 1, 'node at the max depth has multiple rewards!'
                        dummy.backpropagate(selected_node.reward[0], one_node_up)
                        if verbose:  print('after backprop\n', root)
                    else:
                        raise ValueError('Yout MCTS depth is too small!')
                else:
                    next_requests = requests.from_depth(depth)
                    dummy.expand(selected_node, next_requests)
                    if verbose:  print('after expand\n', root)
                    for child_node in selected_node.children:
                        if verbose:  print('start rollout at', hash(tuple(child_node.buses_paths)))
                        rollout_cost = dummy.rollout(deepcopy(child_node.buses_paths),
                                                    SimRequestChain(requests.chain[depth+1:]))
                        child_node.reward = [rollout_cost]
                        dummy.backpropagate(rollout_cost, selected_node)
        return root

    #TODO do the visualization for the Best action merging as well based on the previous tree algorithm
    @log_runtime_and_memory
    def get_best_action(self, parallel=True) -> list[Routing_plan]:
        #this could be smaller than K_MAX!
        actions: list[tuple[list[Routing_plan], float, int]] = [[None for i in range(3)] for i in range(config.K_MAX)]  # tuple[cost, visits]
        if verbose:  print('----------------------------------------------')
        if verbose:  print('getting best action')
        if parallel == True:
            num_cores = os.cpu_count()
            # TODO should this Thread or Process pool execution
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(self._evaluate_tree)
                           for _ in range(config.N_CHAINS)]
                for future in as_completed(futures):
                    evaluated_tree = future.result()
                    for index, mc_node in enumerate(evaluated_tree.root.children):
                        old_cost = actions[index][1] or 0
                        old_visits = actions[index][2] or 0
                        new_visits = old_visits + 1
                        avg_val = sum(rew for rew in mc_node.reward) / len(mc_node.reward)
                        new_cost = (old_cost * old_visits + avg_val) / new_visits
                        actions[index] = (mc_node.buses_paths, new_cost, new_visits)
        else:
            for _ in range(config.N_CHAINS):
                evaluated_tree = self._evaluate_tree()
                for index, mc_node in enumerate(evaluated_tree.root.children):
                    old_cost = actions[index][1] or 0
                    old_visits = actions[index][2] or 0
                    new_visits = old_visits + 1
                    avg_val = sum(rew for rew in mc_node.reward) / len(mc_node.reward)
                    new_cost = (old_cost * old_visits + avg_val) / new_visits
                    actions[index] = (mc_node.buses_paths, new_cost, new_visits)
                if verbose:  self.print_actions(actions) 
        #TODO should we be able to see all of the actions in here?
        if any(x is None for x in actions):
            raise ValueError('Some of the actions are not computed!')
        best_action = min((x for x in actions if x[0]), key=lambda x: x[1])[0]
        return best_action

    def print_actions(self, actions: list[tuple[list[Routing_plan], float, int]]) -> None:
        for routing_plan, cost, visits in actions:
            if verbose:  print(f'plan {hash(tuple(routing_plan))}, cost {cost:.1f}, visits {visits}')
