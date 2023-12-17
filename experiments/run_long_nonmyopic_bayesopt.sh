#!/bin/bash

# Define an array of experiment configurations

myopic_configurations=(
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name schwefel3d"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name braninhoo"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name styblinskitang2d"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name holdertable"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name eggholder"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name bukinn6"
    "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name rosenbrock"
)

nonmyopic_configurations=(
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name schwefel3d --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name braninhoo --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name styblinskitang2d --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name holdertable --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name eggholder --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name bukinn6 --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name rosenbrock --horizon 0 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name schwefel3d --horizon 1 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name braninhoo --horizon 1 --batch-size 8"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name styblinskitang2d --horizon 1 --batch-size 8"
    "--output-dir longrun-bayesopt-filler --optimize --starts 8 --budget 60 --trials 55 --function-name holdertable --horizon 1 --batch-size 8 --seed 1837134"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name eggholder --horizon 1 --batch-size 8 --nworkers 12"
    # "--output-dir longrun-bayesopt --optimize --starts 8 --budget 60 --trials 50 --function-name bukinn6 --horizon 1 --batch-size 8"
    "--output-dir longrun-bayesopt-filler --optimize --starts 8 --budget 60 --trials 45 --function-name rosenbrock --horizon 1 --batch-size 8"
)

# # Loop over the experiment configurations and run each one in the background
# for config in "${myopic_configurations[@]}"; do
#   julia myopic_bayesopt.jl $config
# done

for config in "${nonmyopic_configurations[@]}"; do
  julia nonmyopic_bayesopt.jl $config
done