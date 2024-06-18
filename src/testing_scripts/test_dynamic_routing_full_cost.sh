#!/bin/bash

for i in $(seq 10 14);
do
    python src/Evaluate_routing.py --mode 0 --day $i --truncation_horizon 25 --consider_route_time > results/full_cost/dynamic_results_rollout_$i.txt
    python src/Evaluate_routing.py --mode 1 --day $i --truncation_horizon 25 --consider_route_time > results/full_cost/dynamic_results_greedy_$i.txt
done