using ArgParse
using Distributions
using LinearAlgebra
using Plots
using Sobol
using Random
using CSV
using DataFrames
using Dates
using SharedArrays
using Distributed

addprocs()

@everywhere include("../testfns.jl")
@everywhere include("../rollout.jl")
@everywhere include("../utils.jl")


function parse_command_line(args)
    parser = ArgParseSettings("Myopic Bayesian Optimization CLI")

    @add_arg_table! parser begin
        "--optimize"
            action = :store_true
            help = "If set, the surrogate's hyperparameters will be optimized"
        "--starts"
            action = :store_arg
            help = "Number of random starts for inner policy optimization (default: 16)"
            default = 8
            arg_type = Int
        "--trials"
            action = :store_arg
            help = "Number of trials with a different initial start (default: 50)"
            default = 50
            arg_type = Int
        "--budget"
            action = :store_arg
            help = "Maximum budget for bayesian optimization (default: 15)"
            default = 15
            arg_type = Int
        "--output-dir"
            action = :store_arg
            help = "Output directory for GAPs and model observations"
            required = true
        "--mc-samples"
            action = :store_arg
            help = "Number of Monte Carlo samples for the acquisition function (default: 25)"
            default = 25
            arg_type = Int
        "--horizon"
            action = :store_arg
            help = "Horizon for the rollout (default: 1)"
            default = 1
            arg_type = Int
        "--batch-size"
            action = :store_arg
            help = "Batch size for the rollout (default: 1)"
            default = 8
            arg_type = Int
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end


@everywhere function measure_gap(observations::Vector{T}, fbest::T) where T <: Number
    ϵ = 1e-8
    initial_minimum = observations[1]
    subsequent_minimums = [
        minimum(observations[1:j]) for j in 1:length(observations)
    ]
    numerator = initial_minimum .- subsequent_minimums
    
    if abs(fbest - initial_minimum) < ϵ
        return 1. 
    end
    
    denominator = initial_minimum - fbest
    result = numerator ./ denominator

    for i in 1:length(result)
        if result[i] < ϵ
            result[i] = 0
        end
    end

    return result
end


function create_gap_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Create directory for finished experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    mkpath(dir_name)

    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["$i" for i in 0:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 2),
            Symbol.(col_names)
        )    
    )

    return path_to_csv_file
end


function create_observation_csv_file(
    parent_directory::String,
    child_directory::String,
    csv_filename::String,
    budget::Int
    )
    # Get directory for experiment
    self_filename, extension = splitext(basename(@__FILE__))
    dir_name = parent_directory * "/" * self_filename * "/" * child_directory
    
    # Write the header to the csv file
    path_to_csv_file = dir_name * "/" * csv_filename
    col_names = vcat(["trial"], ["observation_pair_$i" for i in 1:budget])

    CSV.write(
        path_to_csv_file,
        DataFrame(
            -ones(1, budget + 1),
            Symbol.(col_names)
        )    
    )
    
    return path_to_csv_file
end


function write_gap_to_csv(
    gaps::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write gap to csv
    CSV.write(
        path_to_csv_file,
        Tables.table(
            hcat([trial_number gaps'])
        ),
        append=true,
    )

    return nothing
end


function write_observations_to_csv(
    X::Matrix{T},
    y::Vector{T},
    trial_number::Int,
    path_to_csv_file::String
    ) where T <: Number
    # Write observations to csv
    d, N = size(X)
    X = hcat(trial_number * ones(d, 1), X)
    X = vcat(X, [trial_number y'])
    
    CSV.write(
        path_to_csv_file,
        Tables.table(X),
        append=true,
    )

    return nothing
end


function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end


@everywhere function rollout_solver(;
    sur::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64},
    batch::Matrix{Float64},
    max_iterations::Int = 100,
    varred::Bool = true,
    )
    batch_results = Array{Any, 1}(undef, size(batch, 2))

    for i in 1:size(batch, 2)
        # Update start of trajectory for each point in the batch
        tp.x0 = batch[:, i]

        # Perform stochastic gradient ascent on the point in the batch
        batch_results[i] = stochastic_gradient_ascent_adam(
            sur=sur,
            tp=tp,
            max_sgd_iters=max_iterations,
            varred=varred,
            xstarts=xstarts,
        )
    end

    # Find the point in the batch that maximizes the rollout acquisition function
    best_tuple = first(batch_results)
    for result in batch_results[2:end]
        if result.final_obj > best_tuple.final_obj
            best_tuple = result
        end
    end

    return best_tuple.finish, best_tuple.final_obj
end


function distributed_rollout_solver(;
    sur::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64},
    batch::Matrix{Float64},
    max_iterations::Int = 100,
    varred::Bool = true,
    )
    final_locations = SharedMatrix{Float64}(length(tp.x0), size(batch, 2))
    final_evaluations = SharedArray{Float64}(size(batch, 2))

    @sync @distributed for i in 1:size(batch, 2)
        # Update start of trajectory for each point in the batch
        tp.x0 = batch[:, i]

        # Perform stochastic gradient ascent on the point in the batch
        result = stochastic_gradient_ascent_adam(
            sur=sur,
            tp=tp,
            max_sgd_iters=max_iterations,
            varred=varred,
            xstarts=xstarts,
        )
        final_locations[:, i] = result.finish
        final_evaluations[i] = result.final_obj
    end

    # Find the point in the batch that maximizes the rollout acquisition function
    best_ndx, best_evaluation, best_location = 1, first(final_evaluations), final_locations[:, 1]
    for i in 1:size(batch, 2)
        if final_evaluations[i] > best_evaluation
            best_ndx = i
            best_evaluation = final_evaluations[i]
        end
    end

    return (best_location, best_evaluation)
end


@everywhere function write_error_to_disk(filename::String, msg::String)
    # Open a text file in write mode
    open(filename, "w") do file
        # Write a string to the file
        write(file, msg)
    end
end


function main()
    cli_args = parse_command_line(ARGS)
    Random.seed!(2024)
    BUDGET = cli_args["budget"]
    NUMBER_OF_TRIALS = cli_args["trials"]
    NUMBER_OF_STARTS = cli_args["starts"]
    # Create a string of the current time to use as a directory name
    DATA_DIRECTORY = cli_args["output-dir"]
    SHOULD_OPTIMIZE = if haskey(cli_args, "optimize") cli_args["optimize"] else false end
    MC_SAMPLES = cli_args["mc-samples"]
    HORIZON = cli_args["horizon"]
    BATCH_SIZE = cli_args["batch-size"]

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = Dict(
        "gramacylee" => (name="gramacylee", fn=TestGramacyLee, args=()),
        "rastrigin" => (name="rastrigin", fn=TestRastrigin, args=(1)),
        "ackley1d" => (name="ackley1d", fn=TestAckley, args=(1)),
        "ackley2d" => (name="ackley2d", fn=TestAckley, args=(2)),
        "ackley3d" => (name="ackley3d", fn=TestAckley, args=(3)),
        "ackley4d" => (name="ackley4d", fn=TestAckley, args=(4)),
        "ackley10d" => (name="ackley10d", fn=TestAckley, args=(2)),
        "rosenbrock" => (name="rosenbrock", fn=TestRosenbrock, args=()),
        "sixhump" => (name="sixhump", fn=TestSixHump, args=()),
        "braninhoo" => (name="braninhoo", fn=TestBraninHoo, args=()),
        "hartmann3d" => (name="hartmann3d", fn=TestHartmann3D, args=()),
        "goldsteinprice" => (name="goldsteinprice", fn=TestGoldsteinPrice, args=()),
        "beale" => (name="beale", fn=TestBeale, args=()),
        "easom" => (name="easom", fn=TestEasom, args=()),
        "styblinskitang1d" => (name="styblinskitang1d", fn=TestStyblinskiTang, args=(1)),
        "styblinskitang2d" => (name="styblinskitang2d", fn=TestStyblinskiTang, args=(2)),
        "styblinskitang3d" => (name="styblinskitang3d", fn=TestStyblinskiTang, args=(3)),
        "styblinskitang4d" => (name="styblinskitang4d", fn=TestStyblinskiTang, args=(4)),
        "styblinskitang10d" => (name="styblinskitang10d", fn=TestStyblinskiTang, args=(10)),
        "bukinn6" => (name="bukinn6", fn=TestBukinN6, args=()),
        "crossintray" => (name="crossintray", fn=TestCrossInTray, args=()),
        "eggholder" => (name="eggholder", fn=TestEggHolder, args=()),
        "holdertable" => (name="holdertable", fn=TestHolderTable, args=()),
        "schwefel1d" => (name="schwefel1d", fn=TestSchwefel, args=(1)),
        "schwefel2d" => (name="schwefel2d", fn=TestSchwefel, args=(2)),
        "schwefel3d" => (name="schwefel3d", fn=TestSchwefel, args=(3)),
        "schwefel4d" => (name="schwefel4d", fn=TestSchwefel, args=(4)),
        "schwefel10d" => (name="schwefel10d", fn=TestSchwefel, args=(10)),
        "levyn13" => (name="levyn13", fn=TestLevyN13, args=()),
        "trid1d" => (name="trid1d", fn=TestTrid, args=(1)),
        "trid2d" => (name="trid2d", fn=TestTrid, args=(2)),
        "trid3d" => (name="trid3d", fn=TestTrid, args=(3)),
        "trid4d" => (name="trid4d", fn=TestTrid, args=(4)),
        "trid10d" => (name="trid10d", fn=TestTrid, args=(10)),
        "mccormick" => (name="mccormick", fn=TestMccormick, args=()),
        "hartmann6d" => (name="hartmann6d", fn=TestHartmann6D, args=()),
        "hartmann4d" => (name="hartmann4d", fn=TestHartmann4D, args=()),
        "hartmann3d" => (name="hartmann3d", fn=TestHartmann3D, args=()),
    )

    # Gaussian process hyperparameters
    θ, σn2 = [1.], 1e-6
    ψ = kernel_matern52(θ)

    for function_name in keys(testfn_payloads)
        # Build the test function object
        payload = testfn_payloads[function_name]
        println("Running experiment for $(payload.name).")
        testfn = payload.fn(payload.args...)
        lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

        # Generate low discrepancy sequence
        lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON + 1)

        # Allocate initial guesses for optimizer
        initial_guesses = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)

        # Allocate all initial samples
        initial_samples = randsample(NUMBER_OF_TRIALS, testfn.dim, lbs, ubs)

        # Allocate space for GAPS
        rollout_gaps = SharedMatrix{Float64}(NUMBER_OF_TRIALS, BUDGET + 1)
        rollout_observations = Vector{Matrix{Float64}}(undef, NUMBER_OF_TRIALS)

        # Create the CSV for the current test function being evaluated
        rollout_csv_file_path = create_gap_csv_file(DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_gaps.csv", BUDGET)

        # Create the CSV for the current test function being evaluated observations
        rollout_observation_csv_file_path = create_observation_csv_file(
            DATA_DIRECTORY, payload.name, "rollout_h$(HORIZON)_observations.csv", BUDGET
        )

        # Initialize the trajectory parameters
        tp = TrajectoryParameters(
            initial_samples[:, 1], # Will be overriden later
            HORIZON,
            MC_SAMPLES,
            lds_rns,
            lbs,
            ubs,
        )

        # Initialize batch of points to evaluate the rollout acquisition function
        batch = generate_batch(BATCH_SIZE, lbs=tp.lbs, ubs=tp.ubs)

        # TODO: Add the parallelism to number of trials.
        # TODO: Investigate SAA for Optimizer
        @sync @distributed for trial in 1:NUMBER_OF_TRIALS
            try
                println("($(payload.name)) Trial $(trial) of $(NUMBER_OF_TRIALS)...")
                # Initialize surrogate model
                Xinit = initial_samples[:, trial:trial]
                yinit = testfn.f.(eachcol(Xinit))
                sur = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

                # Perform Bayesian optimization iterations
                print("Budget Counter: ")
                for budget in 1:BUDGET
                    # Solve the acquisition function
                    xbest, fbest = rollout_solver(sur=sur, tp=tp, xstarts=initial_guesses, batch=batch)
                    ybest = testfn.f(xbest)
                    # Update the surrogate model
                    sur = update_surrogate(sur, xbest, ybest)

                    if SHOULD_OPTIMIZE
                        sur = optimize_hypers_optim(sur, kernel_matern52)
                    end
                    print("|")
                end
                println()
            catch failure_error
                msg = "($(payload.name)) Trial $(trial) failed with error: $(failure_error)"
                self_filename, extension = splitext(basename(@__FILE__))
                filename = DATA_DIRECTORY * "/" * self_filename * "/" * payload.name * "_failed.txt"
                write_error_to_disk(filename, msg)
            end

            # Compute the GAP of the surrogate model
            fbest = testfn.f(testfn.xopt[1])
            rollout_gaps[trial, :] .= measure_gap(get_observations(sur), fbest)
        end

        for trial in 1:NUMBER_OF_TRIALS
            # Write the GAP to disk
            write_gap_to_csv(rollout_gaps[trial, :], trial, rollout_csv_file_path)

            # Write the surrogate observations to disk
            # write_observations_to_csv(sur.X, get_observations(sur), trial, rollout_observation_csv_file_path)
        end
    end
end

main()