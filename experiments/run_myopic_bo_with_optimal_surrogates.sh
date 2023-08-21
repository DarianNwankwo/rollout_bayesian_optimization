#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--output-dir data-without-optimal-surrogate --function-name --optimize gramacylee"
    "--output-dir data-without-optimal-surrogate --function-name --optimize ackley1d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize sixhump"
    "--output-dir data-without-optimal-surrogate --function-name --optimize braninhoo"
    "--output-dir data-without-optimal-surrogate --function-name --optimize ackley2d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize rosenbrock"
    "--output-dir data-without-optimal-surrogate --function-name --optimize rastrigin"
    "--output-dir data-without-optimal-surrogate --function-name --optimize ackley3d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize ackley4d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize ackley10d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize goldsteinprice"
    "--output-dir data-without-optimal-surrogate --function-name --optimize beale"
    "--output-dir data-without-optimal-surrogate --function-name --optimize easom"
    "--output-dir data-without-optimal-surrogate --function-name --optimize styblinskitank1d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize styblinskitank2d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize styblinskitank3d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize styblinskitank4d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize styblinskitank10d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize bukinn6"
    "--output-dir data-without-optimal-surrogate --function-name --optimize crossintray"
    "--output-dir data-without-optimal-surrogate --function-name --optimize eggholder"
    "--output-dir data-without-optimal-surrogate --function-name --optimize holdertable"
    "--output-dir data-without-optimal-surrogate --function-name --optimize schwefel1d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize schwefel2d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize schwefel3d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize schwefel4d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize schwefel10d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize levyn13"
    "--output-dir data-without-optimal-surrogate --function-name --optimize trid1d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize trid2d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize trid3d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize trid4d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize trid10d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize mccormick"
    "--output-dir data-without-optimal-surrogate --function-name --optimize hartmann3d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize hartmann6d"
    "--output-dir data-without-optimal-surrogate --function-name --optimize hartmann4d"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename --optimize
  julia myopic_bayesopt.jl $config
done