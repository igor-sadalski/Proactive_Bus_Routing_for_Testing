"""Trajectory.py

This module contains all wrapper functions needed to run, visualize, and evaluate trajectories for different routing algorithms.

"""
import optparse
import os
import datetime
import traceback

import pandas as pd

from State import State
from Map_graph import Map_graph
from Request_handler import Request_handler
from Map_graph import Map_graph
from Data_structures import Config_flags, Data_folders, Simulator_config, Date_operational_range, Bus_fleet

from Policies import Proactive_Bus_Routing, Static_Route_Creation_Heuristic, Static_Route_Creation_Rollout

from Baselines import Greedy_dynamic_insertion, Greedy_static_insertion, Greedy_dynamic_insertion_MCTS_baseline, MCTS

def obtain_static_assignment_cost(map_graph: Map_graph, requests_pickup_times, request_capacities, simulator_config: Simulator_config, 
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
                        request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                        actual_pickup_time = real_time + predicted_bus_fleet.routing_plans[bus_index].stops_wait_times[i]
                        wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                        requests_wait_time += wait_time_at_the_station
                        serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
                
                for list_index, dropoff_request_index in enumerate(dropoff_requests_list):
                    if dropoff_request_index != -1:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                        requests_wait_time += wait_time_inside_bus
                        del(serviced_requests[dropoff_request_index])
                
                
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

def obtain_static_assignment_cost(map_graph: Map_graph, requests_pickup_times, request_capacities, simulator_config: Simulator_config, 
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
                        request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                        actual_pickup_time = real_time + predicted_bus_fleet.routing_plans[bus_index].stops_wait_times[i]
                        wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                        requests_wait_time += wait_time_at_the_station
                        serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
                
                for list_index, dropoff_request_index in enumerate(dropoff_requests_list):
                    if dropoff_request_index != -1:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                        requests_wait_time += wait_time_inside_bus
                        del(serviced_requests[dropoff_request_index])
                
                
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

def evaluate_trajectory(map_object: Map_graph, policy_name: str, date_operational_range: Date_operational_range, 
                        data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags,
                        truncation_horizon: int = 50):

    trajectory_cost = 0
    request_handler = Request_handler(data_folders=data_folders,
                                      process_requests=config_flags.process_requests)
    
    state_object = State(map_graph=map_object,
                         date_operational_range=date_operational_range,
                         simulator_config=simulator_config)
    
    new_results_folder = os.path.join(data_folders.dynamic_results_folder, policy_name)
    if not os.path.isdir(new_results_folder):
        os.mkdir(new_results_folder)
    print(new_results_folder)

    data_folders.dynamic_policy_results_folder = new_results_folder
    
    if policy_name == 'proactive_routing_perfect':
        policy_ = Proactive_Bus_Routing(map_graph=map_object,
                                        state_object=state_object,
                                        data_folders=data_folders,
                                        simulator_config=simulator_config,
                                        config_flags=config_flags,
                                        date_operational_range=date_operational_range,
                                        perfect_accuracy=True,
                                        truncation_horizon=truncation_horizon)
    elif policy_name == 'proactive_routing_pred':
        policy_ = Proactive_Bus_Routing(map_graph=map_object,
                                        state_object=state_object,
                                        data_folders=data_folders,
                                        simulator_config=simulator_config,
                                        config_flags=config_flags,
                                        date_operational_range=date_operational_range,
                                        perfect_accuracy=False,
                                        truncation_horizon=truncation_horizon)
    elif policy_name == "greedy":
        policy_ = Greedy_dynamic_insertion(map_graph=map_object,
                                           data_folders=data_folders,
                                           simulator_config=simulator_config,
                                           config_flags=config_flags,)
    elif policy_name == "greedy-PTT":
        policy_ = Greedy_dynamic_insertion_MCTS_baseline(map_graph=map_object,
                                           data_folders=data_folders,
                                           simulator_config=simulator_config,
                                           config_flags=config_flags,
                                           cost_func = 'PTT')
        
    elif policy_name == "greedy-budget":
        policy_ = Greedy_dynamic_insertion_MCTS_baseline(map_graph=map_object,
                                           data_folders=data_folders,
                                           simulator_config=simulator_config,
                                           config_flags=config_flags,
                                           cost_func = 'future requests budget')
        
    elif policy_name == "MCTS-PTT":
        policy_ = MCTS(map_graph=map_object,
                       data_folders=data_folders,
                       simulator_config=simulator_config,
                       config_flags=config_flags,
                       cost_func = 'PTT',
                       request_handler = request_handler)
        
    elif policy_name == "MCTS-budget":
        policy_ = MCTS(map_graph=map_object,
                       data_folders=data_folders,
                       simulator_config=simulator_config,
                       config_flags=config_flags,
                       cost_func = 'future requests budget',
                       request_handler = request_handler)

    
    total_num_requests = 0
    time_step = 0
    initial_requests = request_handler.get_initial_requests(date_operational_range=date_operational_range)
    policy_.assign_requests_and_create_routes(state_object=state_object, 
                                              requests=initial_requests, 
                                              config_flags=config_flags)
    
    hour_range, day_range, month_range, year_range = request_handler.generate_operating_ranges(date_operational_range=date_operational_range)

    for i, hour in enumerate(hour_range):
        for minute in range(60):
            if config_flags.verbose:
                print("Time Step = " + str(time_step))
                print(state_object.step_index)
                print(state_object.bus_stop_index)
                print(state_object.bus_locations)
                print(state_object.bus_fleet)
            minute_online_requests = request_handler.get_online_requests_for_given_minute(date_operational_range=date_operational_range,
                                                                                          year=year_range[i],
                                                                                          month=month_range[i],
                                                                                          day=day_range[i],
                                                                                          hour=hour,
                                                                                          minute=minute)
            
            total_num_requests += len(minute_online_requests)
            policy_.assign_requests_and_create_routes(state_object=state_object, 
                                                       requests=minute_online_requests, 
                                                      config_flags=config_flags)
            for second in range(60):
                state_object.next_state()
            
            time_step += 1
    
    remaining_routes_lengths = []
    for bus_index in range(simulator_config.num_buses):
        current_route_length = len(state_object.bus_fleet.routing_plans[bus_index].route)
        remaining_routes_lengths.append(current_route_length)
    maximum_length_of_route = max(remaining_routes_lengths)

    while maximum_length_of_route > 2:
        empty_df = pd.DataFrame()

        if config_flags.plot_final_routes:
            prev_bus_stops = []
            prev_bus_routes = []
            for bus_index in range(state_object.num_buses):
                current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                prev_bus_routes.append(current_routes)
                prev_bus_stops.append(current_bus_stops)
            policy_.plot_utils.plot_routes_before_assignment_online(map_object=map_object, 
                                                                 requests=empty_df, 
                                                                 prev_bus_stops=prev_bus_stops,
                                                                 prev_bus_routes=prev_bus_routes, 
                                                                 bus_locations=state_object.bus_locations,
                                                                 folder_path=new_results_folder)
        for second in range(60):
                state_object.next_state()
        time_step += 1

        remaining_routes_lengths = []
        for bus_index in range(simulator_config.num_buses):
            current_route_length = len(state_object.bus_fleet.routing_plans[bus_index].route)
            remaining_routes_lengths.append(current_route_length)
        if config_flags.verbose:
            print(state_object.bus_locations)
        maximum_length_of_route = max(remaining_routes_lengths)
    
    print(state_object.request_info)
    final_stage_costs = state_object.stage_costs
    print(final_stage_costs)

    total_route_time = state_object.calculate_total_route_time()
    calc_results = state_object.calculate_average_passenger_wait_time()
    avg_wait_time_at_station, avg_wait_time_on_bus, avg_trip_time, number_of_requests, number_of_passengers = calc_results
    print("Total route time = " + str(total_route_time))
    print("Number of requests serviced = " + str(number_of_requests))
    print("Number of passengers serviced = " + str(number_of_passengers))
    print("Average Passenger Wait Time at the origin station = " + str(avg_wait_time_at_station))
    print("Average Passenger Wait Time inside the bus = " + str(avg_wait_time_on_bus))
    print("Average Passenger Trip Time = " + str(avg_trip_time))

    wait_times_at_station, wait_times_on_bus, trip_times = state_object.calculate_requests_wait_times()

    simulation_cost = state_object.calculate_total_cost(consider_route_time=config_flags.consider_route_time)

    print("Wait times at the station = " + str(wait_times_at_station))
    print("Wait times inside the bus = " + str(wait_times_on_bus))
    print("Trip lengths = " + str(trip_times))

    completed_requests_stats, failed_requests_stats = request_handler.extract_data_statistics_for_given_date(date_operational_range=date_operational_range)

    failed_requests_avg_wait_time = failed_requests_stats.avg_wait_time_for_failed_requests

    count_of_potentially_failed_requests = 0
    for wait_time_at_station in wait_times_at_station:
        if wait_time_at_station > failed_requests_avg_wait_time:
            count_of_potentially_failed_requests += 1
    
    print("Number of requests with higher wait times than the average wait time of real cancelled requests = " + str(count_of_potentially_failed_requests))
    print("Total cost = " + str(simulation_cost))

    trajectory_cost = sum(final_stage_costs)
    print("Trajectory cost with costs from stage costs = " + str(trajectory_cost))
    
    return trajectory_cost

def test_static_allocation(map_object: Map_graph, date_operational_range: Date_operational_range, 
                          data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags,
                          perfect_accuracy: bool = True, truncation_horizon: int = 50) -> None:
    heuristic_start = datetime.datetime.now()
    static_route_creation_heuristic = Static_Route_Creation_Heuristic(map_graph=map_object, 
                                                                      data_folders=data_folders,
                                                                      simulator_config=simulator_config,
                                                                      perfect_accuracy=perfect_accuracy)
    
    assign_result = static_route_creation_heuristic.assign_requests_and_create_routes(date_operational_range=date_operational_range,
                                                                                                                       config_flags=config_flags)
    
    heuristic_total_cost, heuristic_total_bus_cost, heuristic_requests_pickup_times, heuristic_request_capacities = assign_result
    
    heuristic_stop = datetime.datetime.now()
    print("Execution time for heuristic static assignment: " + str(heuristic_stop-heuristic_start))
    print("Cost of our heuristic static assignment based on deltas = " +str(heuristic_total_cost))
    print("Cost of our heuristic static assignment based on deltas per bus = " +str(heuristic_total_bus_cost))
    print("\n")


    rollout_start = datetime.datetime.now()
    static_route_creation_rollout = Static_Route_Creation_Rollout(map_graph=map_object, 
                                                                  data_folders=data_folders,
                                                                  simulator_config=simulator_config,
                                                                  perfect_accuracy=perfect_accuracy)
    
    rollout_total_cost, rollout_total_bus_cost, rollout_requests_info = static_route_creation_rollout.assign_requests_and_create_routes(date_operational_range=date_operational_range,
                                                                                                                 config_flags=config_flags,
                                                                                                                 truncation_horizon=truncation_horizon)
    
    rollout_predicted_bus_fleet, rollout_request_assignment = static_route_creation_rollout.retrieve_all_info()
    rollout_route_cost = obtain_static_assignment_cost(map_graph=map_object,
                                                       requests_pickup_times=rollout_requests_info.requests_pickup_times,
                                                       request_capacities=rollout_requests_info.request_capacities,
                                                       simulator_config=simulator_config,
                                                       predicted_bus_fleet=rollout_predicted_bus_fleet,
                                                       request_assignment=rollout_request_assignment,
                                                       config_flags=config_flags)
    
    print("Number of requests = " + str(len(rollout_requests_info.request_capacities)))

    rollout_stop = datetime.datetime.now()
    print("Execution time for rollout static assignment: " + str(rollout_stop-rollout_start))
    print("Cost of our rollout static assignment based on deltas = " +str(rollout_total_cost))
    print("Cost of our rollout static assignment based on deltas per bus = " +str(rollout_total_bus_cost))
    print("Cost of our rollout static assignment = " +str(sum(rollout_route_cost)))
    print("Cost of our rollout static assignment per bus = " +str(rollout_route_cost))
    if config_flags.verbose:
        print(static_route_creation_rollout.bus_fleet)
    print("\n")

    greedy_start = datetime.datetime.now()
    greedy_static_creation = Greedy_static_insertion(map_graph=map_object, 
                                                     data_folders=data_folders,
                                                     simulator_config=simulator_config,
                                                     perfect_accuracy=perfect_accuracy)
    
    greedy_total_cost, greedy_total_bus_cost, greedy_requests_info = greedy_static_creation.assign_requests_and_create_routes(date_operational_range=date_operational_range,
                                                                                                        config_flags=config_flags)
    
    greedy_predicted_bus_fleet, greedy_request_assignment = greedy_static_creation.retrieve_all_info()
    greedy_route_cost = obtain_static_assignment_cost(map_graph=map_object,
                                                       requests_pickup_times=greedy_requests_info.requests_pickup_times,
                                                       request_capacities=greedy_requests_info.request_capacities,
                                                       simulator_config=simulator_config,
                                                       predicted_bus_fleet=greedy_predicted_bus_fleet,
                                                       request_assignment=greedy_request_assignment,
                                                       config_flags=config_flags)
    
    greedy_stop = datetime.datetime.now()
    print("Execution time for greedy static assignment: " + str(greedy_stop-greedy_start))
    print("Cost of greedy static assignment based on deltas = " +str(greedy_total_cost))
    print("Cost of greedy static assignment based on deltas per bus = " +str(greedy_total_bus_cost))
    print("Cost of greedy static assignment = " +str(sum(greedy_route_cost)))
    print("Cost of greedy static assignment per bus = " +str(greedy_route_cost))
    if config_flags.verbose:
        print(greedy_static_creation.bus_fleet)
    

if __name__ == '__main__':
    """Performs execution delta of the process."""
    pStart = datetime.datetime.now()
    try:
        parser = optparse.OptionParser()
        parser.add_option("--num_buses", type=int, dest='num_buses', default=3, help='Number of buses to be ran in the simulation')
        parser.add_option("--bus_capacity", type=int, dest='bus_capacity', default=20, help='Bus capacity for the buses in the fleet')
        parser.add_option("--year", type=int, dest='year', default=2022, help='Select year of interest.')
        parser.add_option("--month", type=int, dest='month', default=8, help='Select month of interest.')
        parser.add_option("--day", type=int, dest='day', default=17, help='Select day of interest.')
        parser.add_option("--start_hour", type=int, dest='start_hour', default=19, help='Select start hour for the time range of interest')
        parser.add_option("--end_hour", type=int, dest='end_hour', default=2, help='Select end hour for the time range of interest')
        parser.add_option("--truncation_horizon", type=int, dest='truncation_horizon', default=20, help='Number of requests to be considered in rollout planning horizon before truncation.')
        parser.add_option("--consider_route_time", action='store_true', dest='consider_route_time', default=False, help='Flag for considering route time in the cost')
        (options, args) = parser.parse_args()

        num_buses = options.num_buses
        bus_capacity = options.bus_capacity
        perfect_accuracy = True
        date_operational_range = Date_operational_range(year=options.year, 
                                                    month=options.month,
                                                    day=options.day,
                                                    start_hour=options.start_hour,
                                                    end_hour=options.end_hour)
    
        data_folders = Data_folders()
        map_object = Map_graph(initialize_shortest_path=False, 
                               routing_data_folder=data_folders.routing_data_folder,
                               area_text_file=data_folders.area_text_file, 
                               use_saved_map=True, 
                               save_map_structure=False)
        
        simulator_config = Simulator_config(map_object=map_object,
                                        num_buses=num_buses,
                                        bus_capacity=bus_capacity)

        plotting_is_on = True
        config_flags = Config_flags(consider_route_time=options.consider_route_time,
                                include_scaling=False,
                                verbose=False,
                                process_requests=False,
                                plot_initial_routes=plotting_is_on,
                                plot_final_routes=plotting_is_on,
                                create_vid=False)

        #TODO automaticaly save to a file 
        # evaluate_trajectory(map_object, 'greedy-PTT', date_operational_range, 
        #             data_folders, simulator_config, config_flags)
        
        evaluate_trajectory(map_object, 'MCTS-PTT', date_operational_range, 
                            data_folders, simulator_config, config_flags)
         
        # test_static_allocation(map_object=map_object, 
        #                        date_operational_range=date_operational_range,
        #                        data_folders=data_folders,
        #                        simulator_config=simulator_config,
        #                        config_flags=config_flags,
        #                        perfect_accuracy=perfect_accuracy)
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))

