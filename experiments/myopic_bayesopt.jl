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

function poi_solver(s::RBFsurrogate, lbs, ubs; initial_guesses)
    fbest = minimum(get_observations(s))

    function poi(x)
        sx = s(x)
        if sx.σ < 1e-6 return 0 end
        return -cdf(Normal(), (fbest - sx.μ) / sx.σ)
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        result = optimize(poi, lbs, ubs, initial_guess, Fminbox(LBFGS()))
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end
    
    return final_minimizer
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

function ucb_solver(s::RBFsurrogate, lbs, ubs; initial_guesses, β=3.)
    fbest = minimum(get_observations(s))

    function ucb(x)
        sx = s(x)
        return -(sx.μ + β*sx.σ)
    end

    final_minimizer = (initial_guesses[:, 1], Inf)
    
    for j in 1:size(initial_guesses, 2)
        initial_guess = initial_guesses[:, j]
        result = optimize(ucb, lbs, ubs, initial_guess, Fminbox(LBFGS()))
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
    Random.seed!(2024)
    BUDGET = 15
    NUMBER_OF_TRIALS = 50
    NUMBER_OF_STARTS = 64

    # Establish the synthetic functions we want to evaluate our algorithms on.
    testfn_payloads = [
        (name="gramacylee", fn=TestGramacyLee, args=()),
        (name="rastrigin", fn=TestRastrigin, args=(1)),
        (name="ackley1d", fn=TestAckley, args=(1)),
        (name="ackley2d", fn=TestAckley, args=(2)),
        (name="rosenbrock", fn=TestRosenbrock, args=()),
        (name="sixhump", fn=TestSixHump, args=()),
        (name="braninhoo", fn=TestBraninHoo, args=()),
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
        ei_gaps = zeros(NUMBER_OF_TRIALS, BUDGET + 1)
        ucb_gaps = zeros(NUMBER_OF_TRIALS, BUDGET + 1)
        poi_gaps = zeros(NUMBER_OF_TRIALS, BUDGET + 1)

        for trial in 1:NUMBER_OF_TRIALS
            println("Trial $(trial) of $(NUMBER_OF_TRIALS)...")
            # Initialize surrogate model
            Xinit = initial_samples[:, trial:trial]
            yinit = testfn.f.(eachcol(Xinit))
            sur_ei = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)
            sur_poi = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)
            sur_ucb = fit_surrogate(ψ, Xinit, yinit; σn2=σn2)

            # Perform Bayesian optimization iterations
            print("Budget Counter: ")
            for budget in 1:BUDGET
                print("|")
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
            end
            println()

            # Compute the GAP of the surrogate model
            fbest = testfn.f(testfn.xopt[1])
            ei_gaps[trial, :] .= measure_gap(get_observations(sur_ei), fbest)
            ucb_gaps[trial, :] .= measure_gap(get_observations(sur_ucb), fbest)
            poi_gaps[trial, :] .= measure_gap(get_observations(sur_poi), fbest)
        end

        # Create directory for finished experiment
        filename, extension = splitext(basename(@__FILE__))
        dir_name = "plots/" * filename * "/" * payload.name
        mkpath(dir_name)

        # Column names should be the trial number and all the iterations
        col_names = vcat(["trial"], ["$i" for i in 0:BUDGET])

        # Write data in gaps to disk in CSV file format
        CSV.write(
            dir_name * "/ei_gaps.csv",
            DataFrame(
                hcat([t for t in 1:NUMBER_OF_TRIALS], ei_gaps),
                Symbol.(col_names)
            )
        )
        CSV.write(
            dir_name * "/ucb_gaps.csv",
            DataFrame(
                hcat([t for t in 1:NUMBER_OF_TRIALS], ucb_gaps),
                Symbol.(col_names)
            )
        )
        CSV.write(
            dir_name * "/poi_gaps.csv",
            DataFrame(
                hcat([t for t in 1:NUMBER_OF_TRIALS], poi_gaps),
                Symbol.(col_names)
            )
        )
    end

end

main()