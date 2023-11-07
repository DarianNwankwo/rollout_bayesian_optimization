#!/bin/bash

# Define an array of experiment configurations

configurations=(
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name gramacylee"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name ackley1d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name sixhump"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name braninhoo"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name ackley2d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name rosenbrock"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name rastrigin"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name ackley3d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name ackley4d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name ackley10d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name goldsteinprice"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name beale"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name easom"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name styblinskitang1d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name styblinskitang2d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name styblinskitang3d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name styblinskitang4d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name styblinskitang10d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name bukinn6"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name crossintray"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name eggholder"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name holdertable"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name schwefel1d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name schwefel2d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name schwefel3d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name schwefel4d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name schwefel10d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name levyn13"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name trid1d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name trid2d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name trid3d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name trid4d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name trid10d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name mccormick"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name hartmann3d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name hartmann6d"
    "--output-dir data-with-optimal-surrogate --optimize --budget 15 --trials 50 --starts 16 --mc-samples 50 --horizon 1 --function-name hartmann4d"
)

# Loop over the experiment configurations and run each one in the background
for config in "${configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia nonmyopic_bayesopt.jl $config
done