# Parse command line arguments
cli_args = ["HORIZON", "MC_SAMPLES", "BUDGET", "NUM_TRIALS", "TESTFUNC_NAME", "TESTFUNC_ARGS"]
if length(ARGS) != length(cli_args)
    local str_builder = "Usage: julia bayesopt1d.jl "
    for arg in cli_args
        str_builder *= "<$arg> "
    end
    println(str_builder)
    exit(1)
end

using Distributed
using Distributions
using LinearAlgebra
using Plots
using Sobol
using Random

addprocs()
println("Total Workers: $(nworkers())")
exit(1)

include("../rollout.jl")
include("../testfns.jl")
include("./utils.jl")

Random.seed!(1906 + 1867 + 1865)

# Test function mappings
testfns = Dict(
    "gramacy_lee" => TestGramacyLee,
    "rastrigin" => TestRastrigin,
    "ackley" => TestAckley,
)

# Get function name and arguments
func_name = ARGS[5]
func_args = length(ARGS) > 5 ? ARGS[6:end] : []
func_args = length(func_args) >= 1 ? [parse(Int64, arg) for arg in func_args] : []
testfn = testfns[func_name](func_args...)

# Global parameters
MAX_SGD_ITERS = 500
BATCH_SIZE = 4
HORIZON = parse(Int64, ARGS[1])
MC_SAMPLES = parse(Int64, ARGS[2])
BUDGET = parse(Int64, ARGS[3])
NUM_TRIALS = parse(Int64, ARGS[4])

# Setup toy problem
testfn = testfns[func_name](func_args...)
lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

# Get NUM_TRIALS initial samples
initial_samples = randsample(NUM_TRIALS, testfn.dim, lbs, ubs)

# Setup low discrepancy random number stream
lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON+1);

# Surrogate hyperparameters and initial setup
θ, output_var = [1.], 1.
ψ = kernel_scale(kernel_matern52, [output_var, θ...]);

# Generate batch of locations to perform SGA on
batch = generate_batch(BATCH_SIZE; lbs=lbs, ubs=ubs)

for trial in 1:NUM_TRIALS
    # Setup structures for collection trial data
    X = reshape(initial_samples[:, trial], testfn.dim, 1)
    sur = fit_surrogate(ψ, X, testfn.f; σn2=σn2)

    for budget in 1:BUDGET
        # # Optimize each batch location in parallel
        # batch_results = @distributed (append!) for j = 1:size(batch, 2)
        #     # Setup parameters for gradient ascent for each process
        # end

    end
    # Update collective data
end


filename, extension = splitext(basename(@__FILE__))
dirs = [func_name]
create_experiment_directory(filename, dirs)