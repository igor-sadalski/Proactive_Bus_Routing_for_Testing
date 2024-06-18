from Map_graph import Map_graph


K_MAX = 10 #10
NUM_BUSSES = 3
MCTS_DEPTH = 5 #10  
SINGLE_MCTREE_ITERATIONS = 100 #50
MCTS_TUNING_PARAM = 500 #TODO this need to be significnatly larger to have any impacet on the values
SAMPLED_BANK_SIZE = 10000
N_CHAINS = 5 #same as number of MCTrees in MCForest

routing_data_folder="data/routing_data"
area_text_file="data/Evening_Van_polygon.txt"
MAP_GRAPH = Map_graph(initialize_shortest_path=False, 
                           routing_data_folder=routing_data_folder,
                            area_text_file=area_text_file, 
                            use_saved_map=True, 
                            save_map_structure=False)
