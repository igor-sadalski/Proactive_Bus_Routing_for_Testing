#!/bin/bash

for i in $(seq 15 19);
do
    python src/Evaluate_routing.py --mode 0 --day $i --truncation_horizon 25 > results/wait_time/dynamic_results_rollout_$i.txt
    python src/Evaluate_routing.py --mode 1 --day $i --truncation_horizon 25 > results/wait_time/dynamic_results_greedy_$i.txt
done