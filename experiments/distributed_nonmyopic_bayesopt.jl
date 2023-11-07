using ArgParse
using Distributions
using LinearAlgebra
using Plots
using Sobol
using Random
using CSV
using DataFrames
using Dates
using Distributed
using SharedArrays


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
        "--function-name"
            action = :store_arg
            help = "Function name"
            required = true
        "--starts"
            action = :store_arg
            help = "Number of random starts for inner policy optimization (default: 16)"
            default = 16
            arg_type = Int
        "--trials"
            action = :store_arg
            help = "Number of trials with a different initial start (default: 50)"
            default = 10
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


function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end


function rollout_solver(;
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


function main()
    cli_args = parse_command_line(ARGS)
    Random.seed!(1823)
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

    # Build the test function object
    payload = testfn_payloads[cli_args["function-name"]]
    println("Running experiment for $(payload.name) with $(nprocs()) processes...")
    testfn = payload.fn(payload.args...)
    lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

    # Generate low discrepancy sequence
    lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON + 1)

    # Allocate initial guesses for optimizer
    initial_guesses = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)

    # Allocate all initial samples
    initial_samples = randsample(NUMBER_OF_TRIALS, testfn.dim, lbs, ubs)

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

    for trial in 1:NUMBER_OF_TRIALS
        println("($(payload.name)) Trial $(trial) of $(NUMBER_OF_TRIALS)...")
        # Initialize surrogate model
        Xinit = initial_samples[:, trial:trial]
        yinit = testfn.f.(eachcol(Xinit))
        sur = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

        # Perform Bayesian optimization iterations
        print("Budget Counter: ")
        for budget in 1:BUDGET
            # Solve the acquisition function
            xbest, fbest = distributed_rollout_solver(sur=sur, tp=tp, xstarts=initial_guesses, batch=batch)
            # xbest, fbest = rollout_solver(sur=sur, tp=tp, xstarts=initial_guesses, batch=batch)
            ybest = testfn.f(xbest)
            # Update the surrogate model
            sur = update_surrogate(sur, xbest, ybest)

            if SHOULD_OPTIMIZE
                sur = optimize_hypers_optim(sur, kernel_matern52)
            end
            print("|")
        end
        println()
    end 

end

main()