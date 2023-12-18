#!/bin/bash

# Define an array of experiment configurations
nonmyopic_configurations=(
    # Beginning of Horizon 0 Nonmyopic Experiments
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name ackley4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name braninhoo"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name bukinn6"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name easom"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name eggholder"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name goldsteinprice"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name gramacylee"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name hartmann3d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name holdertable"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name levyn13"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name mccormick"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name rastrigin4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name rosenbrock"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name schwefel3d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name schwefel4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name sixhump"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name styblinskitang2d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name trid1d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 0 --function-name trid4d"
    # Beginning of Horizon 1 Nonmyopic Experiments
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name ackley4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name braninhoo"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name bukinn6"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name easom"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name eggholder"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name goldsteinprice"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name gramacylee"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name hartmann3d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name holdertable"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name levyn13"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name mccormick"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name rastrigin4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name rosenbrock"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name schwefel3d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name schwefel4d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name sixhump"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name styblinskitang2d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name trid1d"
    "--output-dir data-with-optimal-surrogate-without-varred --optimize --budget 15 --trials 60 --horizon 1 --function-name trid4d"
)

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia nonmyopic_bayesopt.jl $config
done