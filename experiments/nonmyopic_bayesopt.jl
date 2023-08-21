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


function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end


function ei_solver(s::RBFsurrogate, lbs, ubs; initial_guesses)
    fbest = minimum(get_observations(s))

    function ei(x)
        sx = s(x)
        if sx.σ < 1e-6 return 0 end
        return -sx.EI
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        result = optimize(ei, lbs, ubs, initial_guess, Fminbox(LBFGS()))
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end
    
    return final_minimizer
end


function main()
    Random.seed!(2024)
    BUDGET = 100
    NUMBER_OF_TRIALS = 50
    NUMBER_OF_STARTS = 16
    DATA_DIRECTORY = "test"

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = [
        # (name="gramacylee", fn=TestGramacyLee, args=()),
        # (name="rastrigin", fn=TestRastrigin, args=(1)),
        # (name="ackley1d", fn=TestAckley, args=(1)),
        # (name="ackley2d", fn=TestAckley, args=(2)),
        (name="rosenbrock", fn=TestRosenbrock, args=()),
        # (name="sixhump", fn=TestSixHump, args=()),
        # (name="braninhoo", fn=TestBraninHoo, args=()),
    ]

    # Gaussian process hyperparameters
    θ, σn2 = [1.], 1e-6
    ψ = kernel_matern52(θ)

    for payload in testfn_payloads
        println("Running experiment for $(payload.name)...")
        # Build the test function object
        testfn = payload.fn(payload.args...)
        lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

        # Allocate initial guesses for optimizer
        initial_guesses = generate_initial_guesses(NUMBER_OF_STARTS, lbs, ubs)

        # Allocate all initial samples
        initial_samples = randsample(NUMBER_OF_TRIALS, testfn.dim, lbs, ubs)

        # Allocate space for GAPS
        rollout_gaps = zeros(BUDGET + 1)

        # Create the CSV for the current test function being evaluated
        csv_file_path = create_gap_csv_file(DATA_DIRECTORY, payload.name, "rollout_gaps.csv", BUDGET)

        for trial in 1:NUMBER_OF_TRIALS
            println("Trial $(trial) of $(NUMBER_OF_TRIALS)...")
            # Initialize surrogate model
            Xinit = initial_samples[:, trial:trial]
            yinit = testfn.f.(eachcol(Xinit))
            sur = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

            # Perform Bayesian optimization iterations
            print("Budget Counter: ")
            for budget in 1:BUDGET
                print("|")
                # Solve the acquisition function
                # xbest, fbest = rollout_solver(sur, lbs, ubs; initial_guesses=initial_guesses)
                xbest, fbest = ei_solver(sur, lbs, ubs; initial_guesses=initial_guesses)
                ybest = testfn.f(xbest)
                # Update the surrogate model
                sur = update_surrogate(sur, xbest, ybest)
            end
            println()

            # Compute the GAP of the surrogate model
            fbest = testfn.f(testfn.xopt[1])
            rollout_gaps[:] .= measure_gap(get_observations(sur), fbest)

            # Write the GAP to the CSV file
            write_gap_to_csv(rollout_gaps, trial, csv_file_path)
        end
    end

end

main()