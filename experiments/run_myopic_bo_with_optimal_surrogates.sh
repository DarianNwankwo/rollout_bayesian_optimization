#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--output-dir data-with-optimal-surrogate --optimize --function-name gramacylee"
    "--output-dir data-with-optimal-surrogate --optimize --function-name ackley1d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name sixhump"
    "--output-dir data-with-optimal-surrogate --optimize --function-name braninhoo"
    "--output-dir data-with-optimal-surrogate --optimize --function-name ackley2d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name rosenbrock"
    "--output-dir data-with-optimal-surrogate --optimize --function-name rastrigin"
    "--output-dir data-with-optimal-surrogate --optimize --function-name ackley3d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name ackley4d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name ackley10d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name goldsteinprice"
    "--output-dir data-with-optimal-surrogate --optimize --function-name beale"
    "--output-dir data-with-optimal-surrogate --optimize --function-name easom"
    "--output-dir data-with-optimal-surrogate --optimize --function-name styblinskitang1d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name styblinskitang2d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name styblinskitang3d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name styblinskitang4d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name styblinskitang10d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name bukinn6"
    "--output-dir data-with-optimal-surrogate --optimize --function-name crossintray"
    "--output-dir data-with-optimal-surrogate --optimize --function-name eggholder"
    "--output-dir data-with-optimal-surrogate --optimize --function-name holdertable"
    "--output-dir data-with-optimal-surrogate --optimize --function-name schwefel1d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name schwefel2d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name schwefel3d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name schwefel4d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name schwefel10d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name levyn13"
    "--output-dir data-with-optimal-surrogate --optimize --function-name trid1d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name trid2d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name trid3d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name trid4d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name trid10d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name mccormick"
    "--output-dir data-with-optimal-surrogate --optimize --function-name hartmann3d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name hartmann6d"
    "--output-dir data-with-optimal-surrogate --optimize --function-name hartmann4d"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename --optimize
  julia myopic_bayesopt.jl $config
done