#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--output-dir dwos-filler --optimize --budget 15 --trials 50 --horizon 1 --function-name levyn13 --seed 275"
    "--output-dir dwos-filler --optimize --budget 15 --trials 50 --horizon 1 --function-name goldsteinprice --seed 10096"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia nonmyopic_bayesopt.jl $config
done