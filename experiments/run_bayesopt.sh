#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name gramacylee"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name ackley --function-arg 1"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name sixhump"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name branin"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name ackley --function-arg 2"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name rosenbrock --objective EI"
    "--horizon 0 --mc-samples 50 --budget 15 --num-trials 1 --sgd-iters 100 --batch-size 64 --function-name rastrigin"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia bayesopt1d.jl $config > "$output_file" &
done
