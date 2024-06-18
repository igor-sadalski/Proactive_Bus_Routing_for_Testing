#!/bin/bash

for i in $(seq 15 19);
do
    python src/Trajectory.py --day $i --truncation_horizon 25 > results/wait_time/static_results_$i.txt
done