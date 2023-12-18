#!/bin/bash

# Define an array of experiment configurations

# TODO: Make sure we have configurations for timing myopic bo and nonmyopic bo
myopic_configurations=(
    "--output-dir data-without-optimal-surrogate --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name gramacylee"
)

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia timed_myopic_bo.jl $config
done

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia timed_nonmyopic_bo.jl $config
done