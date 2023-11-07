#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name gramacylee"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name ackley --function-arg 1"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name sixhump"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name braninhoo"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name ackley --function-arg 2"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name rosenbrock --objective EI"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name rastrigin --function-arg 1"
    "--horizon 1 --mc-samples 50 --budget 15 --num-trials 50 --sgd-iters 100 --batch-size 64 --function-name rastrigin --function-arg 2"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia myopic_bayesopt.jl $config &
done