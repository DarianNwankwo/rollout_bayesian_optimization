myopic_configurations=(
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name gramacylee"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name rastrigin4d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name ackley4d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name rosenbrock"
    "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 60 --function-name sixhump"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name braninhoo"
    "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 60 --function-name hartmann3d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name goldsteinprice"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name easom"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name styblinskitang2d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name bukinn6"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name eggholder"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name holdertable"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name schwefel3d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name schwefel4d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name levyn13"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name trid1d"
    # "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 50 --function-name trid4d"
    "--output-dir ../myopic-timing --optimize --starts 8 --budget 20 --trials 60 --function-name mccormick"
)

for config in "${myopic_configurations[@]}"; do
  julia ../timed_myopic_bo.jl $config
done