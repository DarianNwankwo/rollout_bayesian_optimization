using ArgParse
using Distributions
using LinearAlgebra
using Plots
using Sobol
using Random
using CSV
using DataFrames


include("../testfns.jl")
include("../utils.jl")
include("../rollout.jl")


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
    end

    parsed_args = parse_args(args, parser)
    return parsed_args
end

"""
Given a limited evaluation budget, we evaluate the performance of an algorithm in terms of gap G. The gap measures the
best decrease in objective function from the first to the last iteration, normalized by the maximum reduction possible.
"""
function measure_gap(observations::Vector{T}, fbest::T) where T <: Number
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

"""
Creates a csv file to store the observations of the algorithm.
"""
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

function poi_solver(s::RBFsurrogate, lbs, ubs; initial_guesses, max_iterations=100)
    fbest = minimum(get_observations(s))

    function poi(x)
        sx = s(x)
        if sx.σ < 1e-6 return 0 end
        return -cdf(Normal(), (fbest - sx.μ) / sx.σ)
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        result = optimize(poi, lbs, ubs, initial_guess, Fminbox(LBFGS()), Optim.Options(iterations=max_iterations))
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end
    
    return final_minimizer
end

"""
TODO: We need to specify the maximum number of iterations and terminate if we exhaust our budget
TODO: EI for Rosenbrock looks like zeros everywhere, depending on how we sample. I suspect this
is why our algorithm halts here.
"""
function ei_solver(s::RBFsurrogate, lbs, ubs; initial_guesses, max_iterations=100)
    fbest = minimum(get_observations(s))

    function ei(x)
        sx = s(x)
        if sx.σ < 1e-6 return 0 end
        return -sx.EI
    end

    function ei_grad!(g, x)
        EIx = -s(x).∇EI
        for i in eachindex(EIx)
            g[i] = EIx[i]
        end
    end

    function ei_hessian!(h, x)
        HEIx = -s(x).HEI
        for row in 1:size(HEIx, 1)
            for col in 1:size(HEIx, 2)
                h[row, col] = HEIx[row, col]
            end
        end
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        df = TwiceDifferentiable(ei, ei_grad!, ei_hessian!, initial_guess)
        dfc = TwiceDifferentiableConstraints(lbs, ubs)
        result = optimize(df, dfc, initial_guess, IPNewton(), Optim.Options(iterations=max_iterations))
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end
    
    return final_minimizer
end

function ucb_solver(s::RBFsurrogate, lbs, ubs; initial_guesses, β=3., max_iterations=100)
    fbest = minimum(get_observations(s))

    function ucb(x)
        sx = s(x)
        return -(sx.μ + β*sx.σ)
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        result = optimize(ucb, lbs, ubs, initial_guess, Fminbox(LBFGS()), Optim.Options(iterations=max_iterations))
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end
    
    return final_minimizer
end

function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end

function main()
    cli_args = parse_command_line(ARGS)
    Random.seed!(2024)
    BUDGET = cli_args["budget"]
    NUMBER_OF_TRIALS = cli_args["trials"]
    NUMBER_OF_STARTS = cli_args["starts"]
    DATA_DIRECTORY = cli_args["output-dir"]
    SHOULD_OPTIMIZE = if haskey(cli_args, "optimize") cli_args["optimize"] else false end

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
    )

    # Gaussian process hyperparameters
    θ, σn2 = [1.], 1e-6
    ψ = kernel_matern52(θ)

    # Build the test function object
    payload = testfn_payloads[cli_args["function-name"]]
    println("Running experiment for $(payload.name)...")
    testfn = payload.fn(payload.args...)
    lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

    # Allocate initial guesses for optimizer
    initial_guesses = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)

    # Allocate all initial samples
    initial_samples = randsample(NUMBER_OF_TRIALS, testfn.dim, lbs, ubs)

    # Allocate space for GAPS
    ei_gaps = zeros(BUDGET + 1)
    ucb_gaps = zeros(BUDGET + 1)
    poi_gaps = zeros(BUDGET + 1)

    # Create the CSV for the current test function being evaluated
    ei_csv_file_path = create_gap_csv_file(DATA_DIRECTORY, payload.name, "ei_gaps.csv", BUDGET)
    ucb_csv_file_path = create_gap_csv_file(DATA_DIRECTORY, payload.name, "ucb_gaps.csv", BUDGET)
    poi_csv_file_path = create_gap_csv_file(DATA_DIRECTORY, payload.name, "poi_gaps.csv", BUDGET)

    # Create the CSV for the current test function being evaluated observations
    ei_observation_csv_file_path = create_observation_csv_file(
        DATA_DIRECTORY, payload.name, "ei_observations.csv", BUDGET
    )
    ucb_observation_csv_file_path = create_observation_csv_file(
        DATA_DIRECTORY, payload.name, "ucb_observations.csv", BUDGET
    )
    poi_observation_csv_file_path = create_observation_csv_file(
        DATA_DIRECTORY, payload.name, "poi_observations.csv", BUDGET
    )

    for trial in 1:NUMBER_OF_TRIALS
        println("($(payload.name)) Trial $(trial) of $(NUMBER_OF_TRIALS)...")
        # Initialize surrogate model
        Xinit = initial_samples[:, trial:trial]
        yinit = testfn.f.(eachcol(Xinit))
        sur_ei = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)
        sur_poi = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)
        sur_ucb = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

        # Perform Bayesian optimization iterations
        print("Budget Counter: ")
        for budget in 1:BUDGET
            # Solve the acquisition function
            xbest, fbest = poi_solver(sur_poi, lbs, ubs; initial_guesses=initial_guesses)
            ybest = testfn.f(xbest)
            # Update the surrogate model
            sur_poi = update_surrogate(sur_poi, xbest, ybest)

            # Solve the acquisition function
            xbest, fbest = ei_solver(sur_ei, lbs, ubs; initial_guesses=initial_guesses)
            ybest = testfn.f(xbest)
            # Update the surrogate model
            sur_ei = update_surrogate(sur_ei, xbest, ybest)
            
            # Solve the acquisition function
            xbest, fbest = ucb_solver(sur_ucb, lbs, ubs; initial_guesses=initial_guesses)
            ybest = testfn.f(xbest)
            # Update the surrogate model
            sur_ucb = update_surrogate(sur_ucb, xbest, ybest)

            if SHOULD_OPTIMIZE
                sur_poi = optimize_hypers_optim(sur_poi, kernel_matern52)
                sur_ei = optimize_hypers_optim(sur_ei, kernel_matern52)
                sur_ucb = optimize_hypers_optim(sur_ucb, kernel_matern52)
            end
            print("|")
        end
        println()

        # Compute the GAP of the surrogate model
        fbest = testfn.f(testfn.xopt[1])
        ei_gaps[:] .= measure_gap(get_observations(sur_ei), fbest)
        ucb_gaps[:] .= measure_gap(get_observations(sur_ucb), fbest)
        poi_gaps[:] .= measure_gap(get_observations(sur_poi), fbest)

        # Write the GAP to disk
        write_gap_to_csv(ei_gaps, trial, ei_csv_file_path)
        write_gap_to_csv(ucb_gaps, trial, ucb_csv_file_path)
        write_gap_to_csv(poi_gaps, trial, poi_csv_file_path)

        # Write the surrogate observations to disk
        write_observations_to_csv(sur_ei.X, get_observations(sur_ei), trial, ei_observation_csv_file_path)
        write_observations_to_csv(sur_ucb.X, get_observations(sur_ucb), trial, ucb_observation_csv_file_path)
        write_observations_to_csv(sur_poi.X, get_observations(sur_poi), trial, poi_observation_csv_file_path)
    end 

end

main()