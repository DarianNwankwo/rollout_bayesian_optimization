#!/bin/bash

# Define an array of experiment configurations for specific test functions
nonmyopic_configurations=(
    # Selected Nonmyopic Experiments for Horizons 0 to 2 wihtout Variance Reduction
    # Horizon 0
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=braninhoo --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=bukinn6 --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=eggholder --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=holdertable --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=rosenbrock --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=schwefel3d --horizon 0"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=styblinskitang2d --horizon 0"
    # Horizon 1
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=braninhoo --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=bukinn6 --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=eggholder --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=holdertable --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=rosenbrock --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=schwefel3d --horizon 1"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=styblinskitang2d --horizon 1"
    # Horizon 2
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=braninhoo --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=bukinn6 --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=eggholder --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=holdertable --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=rosenbrock --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=schwefel3d --horizon 2"
    "--output-dir=../nonmyopic-longrun-gaps-and-time --optimize --budget 80 --function-name=styblinskitang2d --horizon 2"
    # Selected Nonmyopic Experiments with Variance Reduction
    # Horizon 0 with Variance Reduction
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=braninhoo --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=bukinn6 --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=eggholder --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=holdertable --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=rosenbrock --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=schwefel3d --horizon 0"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=styblinskitang2d --horizon 0"
    # Horizon 1 with Variance Reduction
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=braninhoo --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=bukinn6 --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=eggholder --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=holdertable --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=rosenbrock --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=schwefel3d --horizon 1"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=styblinskitang2d --horizon 1"
    # Horizon 2 with Variance Reduction
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=braninhoo --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=bukinn6 --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=eggholder --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=holdertable --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=rosenbrock --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=schwefel3d --horizon 2"
    "--output-dir=../nonmyopic-longrun-with-variance-reduction-gaps-and-time --optimize --budget 80 --variance-reduction --function-name=styblinskitang2d --horizon 2"
)

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia ../nonmyopic_bayesopt.jl $config &
done
