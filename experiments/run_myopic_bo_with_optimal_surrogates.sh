#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--output-dir data-without-optimal-surrogate --function-name gramacylee"
    "--output-dir data-without-optimal-surrogate --function-name ackley1d"
    "--output-dir data-without-optimal-surrogate --function-name sixhump"
    "--output-dir data-without-optimal-surrogate --function-name braninhoo"
    "--output-dir data-without-optimal-surrogate --function-name ackley2d"
    "--output-dir data-without-optimal-surrogate --function-name rosenbrock"
    "--output-dir data-without-optimal-surrogate --function-name rastrigin"
    "--output-dir data-without-optimal-surrogate --function-name ackley3d"
    "--output-dir data-without-optimal-surrogate --function-name ackley4d"
    "--output-dir data-without-optimal-surrogate --function-name ackley10d"
    "--output-dir data-without-optimal-surrogate --function-name hartmann3d"
    "--output-dir data-without-optimal-surrogate --function-name goldsteinprice"
    "--output-dir data-without-optimal-surrogate --function-name beale"
    "--output-dir data-without-optimal-surrogate --function-name easom"
    "--output-dir data-without-optimal-surrogate --function-name styblinskitank1d"
    "--output-dir data-without-optimal-surrogate --function-name styblinskitank2d"
    "--output-dir data-without-optimal-surrogate --function-name styblinskitank3d"
    "--output-dir data-without-optimal-surrogate --function-name styblinskitank4d"
    "--output-dir data-without-optimal-surrogate --function-name styblinskitank10d"
    "--output-dir data-without-optimal-surrogate --function-name bukinn6"
    "--output-dir data-without-optimal-surrogate --function-name crossintray"
    "--output-dir data-without-optimal-surrogate --function-name eggholder"
    "--output-dir data-without-optimal-surrogate --function-name holdertable"
    "--output-dir data-without-optimal-surrogate --function-name schwefel1d"
    "--output-dir data-without-optimal-surrogate --function-name schwefel2d"
    "--output-dir data-without-optimal-surrogate --function-name schwefel3d"
    "--output-dir data-without-optimal-surrogate --function-name schwefel4d"
    "--output-dir data-without-optimal-surrogate --function-name schwefel10d"
    "--output-dir data-without-optimal-surrogate --function-name levyn13"
    "--output-dir data-without-optimal-surrogate --function-name trid1d"
    "--output-dir data-without-optimal-surrogate --function-name trid2d"
    "--output-dir data-without-optimal-surrogate --function-name trid3d"
    "--output-dir data-without-optimal-surrogate --function-name trid4d"
    "--output-dir data-without-optimal-surrogate --function-name trid10d"
    "--output-dir data-without-optimal-surrogate --function-name mccormick"
    "--output-dir data-without-optimal-surrogate --function-name hartmann3d"
    "--output-dir data-without-optimal-surrogate --function-name hartmann6d"
    "--output-dir data-without-optimal-surrogate --function-name hartmann4d"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia myopic_bayesopt.jl $config
done