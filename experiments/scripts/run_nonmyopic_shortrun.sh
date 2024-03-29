#!/bin/bash

# Define an array of experiment configurations without variance reduction
nonmyopic_configurations=(
    #################################################################################
    #################### VARIANCE REDUCTION BASED EXPERIMENTS ######################
    #################################################################################
    # Beginning of Horizon 0 Nonmyopic Experiments
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=gramacylee --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rastrigin1d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rastrigin4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley1d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley2d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley3d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley10d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rosenbrock --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=sixhump --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=braninhoo --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann3d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=goldsteinprice --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=beale --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=easom --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang1d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang2d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang3d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang10d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=bukinn6 --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=crossintray --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=eggholder --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=holdertable --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel1d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel2d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel3d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel10d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=levyn13 --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid1d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid2d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid3d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid10d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=mccormick --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann6d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann4d --horizon 0"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann3d --horizon 0"
    # Beginning of Horizon 1 Nonmyopic Experiments
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=gramacylee --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rastrigin1d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rastrigin4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley1d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley2d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley3d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=ackley10d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=rosenbrock --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=sixhump --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=braninhoo --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann3d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=goldsteinprice --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=beale --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=easom --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang1d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang2d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang3d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=styblinskitang10d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=bukinn6 --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=crossintray --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=eggholder --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=holdertable --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel1d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel2d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel3d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=schwefel10d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=levyn13 --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid1d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid2d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid3d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=trid10d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=mccormick --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann6d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann4d --horizon 1"
    "--output-dir=../nonmyopic-shortrun-with-variance-reduction-gaps-and-time --optimize --variance-reduction --function-name=hartmann3d --horizon 1"
)

# Loop over the experiment configurations and run each one in the background
for config in "${nonmyopic_configurations[@]}"; do
  # Run the Julia code in the background with the given configuration and output filename
  julia ../nonmyopic_bayesopt.jl $config
done