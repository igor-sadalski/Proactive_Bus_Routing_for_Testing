#!/usr/bin/python3
import os
import optparse
import datetime
import traceback

from Map_graph import Map_graph

from Trajectory import evaluate_trajectory

from Data_structures import Config_flags, Data_folders, Simulator_config, Date_operational_range


def create_video(policy_name : str = 'base_policy'):
    os.system('ffmpeg -r 1 -start_number 0 -i data/results/'+policy_name+'/frame%0d.png -pix_fmt yuvj420p -vcodec mjpeg -f mov data/results/'+policy_name+'/trajectory.mov')

def evaluate_all_policies(map_object: Map_graph, policy_names: str, date_operational_range: Date_operational_range, 
                          data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags) -> None:
    for key in policy_names.keys():
        policy_name=policy_names[key]
        evaluate_trajectory(map_object=map_object, 
                            policy_name=policy_name,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags)
        if config_flags.create_vid:
            create_video(policy_name=policy_names[key])

def run_experiment(options=None) -> None:
    
    date_operational_range = Date_operational_range(year=options.year, 
                                                    month=options.month,
                                                    day=options.day,
                                                    start_hour=options.start_hour,
                                                    end_hour=options.end_hour)
    
    data_folders = Data_folders()
    
    policy_names = {0:"proactive_routing_perfect",
                    1: "greedy"}
    
    map_object = Map_graph(initialize_shortest_path=options.init_shortest_path, 
                           routing_data_folder=data_folders.routing_data_folder,
                            area_text_file=data_folders.area_text_file, 
                            use_saved_map=options.use_saved_map, 
                            save_map_structure=options.save_map)


    simulator_config = Simulator_config(map_object=map_object,
                                        num_buses=options.num_buses,
                                        bus_capacity=options.bus_capacity)
    
    config_flags = Config_flags(consider_route_time=options.consider_route_time,
                                include_scaling=options.include_scaling,
                                verbose=options.verbose,
                                process_requests=options.process_requests,
                                plot_initial_routes=options.plot_initial_routes,
                                plot_final_routes=options.plot_final_routes,
                                create_vid=options.create_vid)

    if options.mode < 5:
        policy_name=policy_names[options.mode]
        evaluate_trajectory(map_object=map_object, 
                            policy_name=policy_name,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags,
                            truncation_horizon=options.truncation_horizon)
    
    elif options.mode == 5:
        evaluate_all_policies(map_object=map_object, 
                            policy_names=policy_names,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags)
        
        
    else:
        print("Wrong mode selected. Please try again...")

def main() -> int:
    ##############################################
    # Main function, Options
    ##############################################
   
    parser = optparse.OptionParser()
    parser.add_option("--use_saved_map", action='store_true', dest='use_saved_map', default=True, help='Flag for using saved map data instead of populating new structure')
    parser.add_option("--save_map", action='store_true', dest='save_map', default=False, help='Flag for saving the map structure')
    parser.add_option("--verbose", action='store_true', dest='verbose', default=False, help='Flag for printing additional route information')
    parser.add_option("--create_vid", action='store_true', dest='create_vid', default=False, help='Flag for generating video after plotting')
    parser.add_option("--init_shortest_path", action='store_true', dest='init_shortest_path', default=False, help='Flag for initializing pairwise shortest path distances in the map_graph structure')
    parser.add_option("--plot_initial_routes", action='store_true', dest='plot_initial_routes', default=False, help='Flag for graphing the initial bus routes')
    parser.add_option("--plot_final_routes", action='store_true', dest='plot_final_routes', default=False, help='Flag for graphing the final bus routes')
    parser.add_option("--process_requests", action='store_true', dest='process_requests', default=False, help='Flag for processing requests before loading them')
    parser.add_option("--consider_route_time", action='store_true', dest='consider_route_time', default=False, help='Flag for considering route time in the cost')
    parser.add_option("--include_scaling", action='store_true', dest='include_scaling', default=False, help='Flag for considering scaling coefficient for the wait time cost calculation')
    parser.add_option("--approximate", action='store_true', dest='approximate', default=False, help='Flag for approximating request insertion cost')
    parser.add_option("--mode", type=int, dest='mode', default=0, help='Select mode of operation.')
    parser.add_option("--num_buses", type=int, dest='num_buses', default=3, help='Number of buses to be ran in the simulation')
    parser.add_option("--truncation_horizon", type=int, dest='truncation_horizon', default=20, help='Number of requests to be considered in rollout planning horizon before truncation.')
    parser.add_option("--bus_capacity", type=int, dest='bus_capacity', default=20, help='Bus capacity for the buses in the fleet')
    parser.add_option("--year", type=int, dest='year', default=2022, help='Select year of interest.')
    parser.add_option("--month", type=int, dest='month', default=8, help='Select month of interest.')
    parser.add_option("--day", type=int, dest='day', default=17, help='Select day of interest.')
    parser.add_option("--start_hour", type=int, dest='start_hour', default=19, help='Select start hour for the time range of interest') #19
    parser.add_option("--end_hour", type=int, dest='end_hour', default=2, help='Select end hour for the time range of interest')
    (options, args) = parser.parse_args()
    ##############################################
    # Main
    ##############################################
    run_experiment(options)
    return 0


if __name__ == '__main__':
    """Performs execution delta of the process."""
    pStart = datetime.datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))