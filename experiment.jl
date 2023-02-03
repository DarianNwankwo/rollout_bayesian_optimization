include("rollout.jl")

"""
This struct represents an experiment.  It contains all the information needed
to run an experiment, including the functions that define the experiment, the
parameters of the experiment, the data collected so far, and the bounds of the
experiment.
"""
struct Experiment
    f::Function
    ∇f::Function
    ψconstructor::Function
    θ::Vector{Float64}
    y::Vector{Float64}
    X::Matrix{Float64}
    bounds::Matrix{Float64}
    budget::Int64
    horizon::Int64
    sgd_iterations::Int64
    batch_size::Int64
    mc_iterations::Int64
end

function Experiment(;
    f::Function,
    ∇f::Function,
    ψconstructor::Function,
    θ::Vector{Float64},
    y::Vector{Float64},
    X::Matrix{Float64},
    bounds::Matrix{Float64},
    budget::Int64,
    horizon::Int64,
    sgd_iterations::Int64,
    batch_size::Int64,
    mc_iterations::Int64)
    return Experiment(f, ∇f, ψconstructor, θ, y, X, bounds, budget, horizon, sgd_iterations, batch_size, mc_iterations)
end


function run_rollout(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

"""
Study the minimum observed value of the function f over the horizon for some
given budget.
"""
function study_minimum_observed(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

function study_variance_reduction(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

function study_estimation_error(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

function study_optimality_gap(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

function study_model_misspecification(e::Experiment)
    println("Experiment started...")

    println("Experiment completed...")
end

(e::Experiment)() = run(e)

# Given all of the information required to create an Experiment
# write a function that creates an Experiment
function create_experiment()
    # Define the functions that define the experiment
    f(x) = x[1]^2 + x[2]^2
    g(x) = x[1] + x[2]
    psi(x) = x[1] + x[2]

    # Define the parameters of the experiment
    theta = [1.0, 1.0]
    y = [1.0]
    X = [1.0 1.0]
    bounds = [0.0 1.0; 0.0 1.0]
    budget = 10
    horizon = 10
    sgd_iterations = 10
    batch_size = 10
    mc_iterations = 10

    # Create the experiment
    e = Experiment(
        f=f,
        g=g,
        psi=psi,
        theta=theta,
        y=y,
        X=X,
        bounds=bounds,
        budget=budget,
        horizon=horizon,
        sgd_iterations=sgd_iterations,
        batch_size=batch_size,
        mc_iterations=mc_iterations
    )

    return e
end