using ArgParse

function parse_command_line(args)
    parser = ArgParseSettings("Bayesian Optimization CLI")

    @add_arg_table! parser begin
        "--horizon"
            action = :store_arg
            help = "Horizon (default: 0)"
            default = 0
            arg_type = Int
        "--mc-samples"
            action = :store_arg
            help = "Monte Carlo samples (default: 25)"
            default = 25
            arg_type = Int
        "--budget"
            action = :store_arg
            help = "Budget (default: 15)"
            default = 15
            arg_type = Int
        "--num-trials"
            action = :store_arg
            help = "Number of trials (default: 15)"
            default = 15
            arg_type = Int
        "--sgd-iters"
            action = :store_arg
            help = "SGD iterations (default: 100)"
            default = 100
            arg_type = Int
        "--batch-size"
            action = :store_arg
            help = "Batch size (default: 32)"
            default = 32
            arg_type = Int
        "--function-name"
            action = :store_arg
            help = "Function name"
            required = true
        "--function-args"
            action = :store_arg
            help = "Function args"
            default = nothing
        "--objective"
            action = :store_arg
            help = "Objective (default: ei)"
            default = "ei"
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end

cli_args = parse_command_line(ARGS)

using Measures
using Distributed
using Distributions
using LinearAlgebra
using Plots
using SharedArrays
using Sobol
using Random
using CSV
using Tables
using DataFrames

addprocs(4)
println("Total Workers: $(nworkers())")

@everywhere include("../rollout.jl")
@everywhere include("../testfns.jl")
@everywhere include("./utils.jl")
# include("../rollout.jl")
# include("../testfns.jl")
# include("./utils.jl")

Random.seed!(1906 + 1867 + 1865)

# Test function mappings
testfns = Dict(
    "gramacylee" => TestGramacyLee,
    "rastrigin" => TestRastrigin,
    "ackley" => TestAckley,
    "rosenbrock" => TestRosenbrock,
    "sixhump" => TestSixHump,
    "braninhoo" => TestBraninHoo,
)

# Get function name and arguments
# fargs = isnothing(cli_args["function-args"]) ? [] : split(cli_args["function-args"], ",")
# fargs = length(fargs) > 0 ? map(eacharg -> parse(Float64, eacharg), fargs) : []
func_name = cli_args["function-name"]
func_args = isnothing(cli_args["function-args"]) ? [] : split(cli_args["function-args"], ",")
func_args = length(func_args) > 0 ? map(eacharg -> parse(Int64, eacharg), func_args) : []

# Global parameters
MAX_SGD_ITERS = cli_args["sgd-iters"]
BATCH_SIZE = cli_args["batch-size"]
HORIZON = cli_args["horizon"]
MC_SAMPLES = cli_args["mc-samples"]
BUDGET = cli_args["budget"]
NUM_TRIALS = cli_args["num-trials"]
OBJECTIVE = cli_args["objective"] == "ei" ? :EI : :LOGEI

# Setup toy problem
testfn = testfns[func_name](func_args...)
lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]
fbest = testfn.f(first(testfn.xopt))

# Get NUM_TRIALS initial samples
initial_samples = randsample(NUM_TRIALS, testfn.dim, lbs, ubs)

# Setup low discrepancy random number stream
lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON+1);

# Surrogate hyperparameters and initial setup
θ, output_var, σn2 = [1.], 1., 1e-4
ψ = kernel_scale(kernel_matern52, [output_var, θ...]);

# Generate batch of locations to perform SGA on
batch = generate_batch(BATCH_SIZE; lbs=lbs, ubs=ubs)
gaps = zeros(NUM_TRIALS, BUDGET+2) # +2 for initial gap and row index
gaps[:, 1] = cumsum(ones(NUM_TRIALS))

for trial in 1:NUM_TRIALS
    # Setup structures for collection trial data
    X = reshape(initial_samples[:, trial], testfn.dim, 1)
    sur = fit_surrogate(ψ, X, testfn.f; σn2=σn2) # (TODO): Learn kernel hyperparameters

    println("$trial.) Beginning Bayesian Optimization Main Loop")
    println("-----------------------------------------")
    for budget in 1:BUDGET
        println("Iteration #$budget")
        # Optimize each batch location in parallel
        results = []

        results = @distributed (append!) for j = 1:size(batch, 2)
            x0 = batch[:, j]

            res = stochastic_gradient_ascent_adam(x0;
                max_sgd_iters=MAX_SGD_ITERS, lbs=lbs, ubs=ubs, mc_iters=MC_SAMPLES, objective=OBJECTIVE,
                lds_rns=lds_rns, horizon=HORIZON, sur=sur, gtol=1e-10, ftol=1e-8, max_counter=10
            )

            [res]
        end # END @distributed

        # @sync @distributed for j = 1:size(batch, 2)
        #     x0 = batch[:, j]

        #     res = stochastic_gradient_ascent_adam(x0;
        #         max_sgd_iters=MAX_SGD_ITERS, lbs=lbs, ubs=ubs, mc_iters=MC_SAMPLES,
        #         lds_rns=lds_rns, horizon=HORIZON, sur=sur, gtol=1e-10, ftol=1e-8, max_counter=10
        #     )
        # end # END @distributed

        # for j = 1:size(batch, 2)
        #     x0 = batch[:, j]

        #     res = stochastic_gradient_ascent_adam(x0;
        #         max_sgd_iters=MAX_SGD_ITERS, lbs=lbs, ubs=ubs, mc_iters=MC_SAMPLES,
        #         lds_rns=lds_rns, horizon=HORIZON, sur=sur, gtol=1e-10, ftol=1e-8, max_counter=10
        #     )
        #     push!(results, res)
        # end
        
        if length(results) == 0
            start = finish = rand(testfn.dim) .* (ubs - lbs) + lbs
            push!(results, (start=start, finish=finish, final_obj=nothing, final_grad=nothing, iters=0, success=true))
        end

        # Update surrogate with element that optimize the acquisition function
        max_ndx = findmax(t -> t.final_obj, results)[2]
        max_pairing = results[max_ndx]
        xnew = max_pairing.finish
        native_y = recover_y(sur)

        sur = fit_surrogate(ψ, hcat(sur.X, xnew), vcat(native_y, testfn.f(xnew)); σn2=σn2)
        res = optimize_hypers_optim(sur, kernel_matern52; σn2=σn2)
        σ, ℓ = Optim.minimizer(res)
        global ψ = kernel_scale(kernel_matern52, [σ, ℓ])
        native_y = recover_y(sur)
        sur = fit_surrogate(ψ, sur.X, native_y; σn2=σn2)
    end # END Bayesian Optimization Loop
    println("-----------------------------------------")
    
    # Update collective data
    gaps[trial, 2:end] = measure_gap(sur, fbest)
    println("\nObservations: $(sur.y) -- True Best: $fbest")
end


filename, extension = splitext(basename(@__FILE__))
dirs = [func_name]
dir_name = create_experiment_directory(filename, dirs)

csv_filename = create_filename(cli_args)
csv_headers = vcat(["trial_number"], ["budget_$i" for i in 1:BUDGET+1])
CSV.write(
    string(dir_name, "/", csv_filename),
    Tables.table(gaps),
    header=csv_headers,
    writeheader=true
)