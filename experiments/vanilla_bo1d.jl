# Parse command line arguments
args = ARGS
if length(args) == 0
    println("Error: function name is missing")
    exit(1)
end


using Distributions
using LinearAlgebra
using Plots
using Test
using Sobol
using Optim
using Profile
using PProf
using Random

include("../rollout.jl")
include("../testfns.jl")

# Test function mappings
testfns = Dict(
    "gramacy_lee" => TestGramacyLee,
    "rastrigin" => TestRastrigin,
    "ackley" => TestAckley,
)
initial_samples = Dict(
    "gramacy_lee" => [1.],
    "rastrigin" => [1.],
    "ackley" => [-20.],
)

# Get function name and arguments
func_name = args[1]
func_args = length(args) > 1 ? args[2:end] : []
func_args = length(func_args) >= 1 ? [parse(Int64, arg) for arg in func_args] : []
testfn = testfns[func_name](func_args...)

println("Testing function: $(func_name)($(func_args)) -- testfn.dim = $(testfn.dim)")

# Global parameters
MAX_SGD_ITERS = 500
BATCH_SIZE = 32
HORIZON = 0
MC_SAMPLES = 25
BUDGET = 15

# Setup toy problem
testfn = testfns[fn_name](fn_args...)
lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

# Setup low discrepancy random number stream
lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON+1);
# rns = randn(MC_SAMPLES, testfn.dim+1, HORIZON+1);

# Gather initial samples/experimental data
N, θ = 1, [.25]
X = initial_samples[func_name]
X = reshape(X, 1, length(X))
ψ = kernel_scale(kernel_matern52, [1., θ...]);

# Define the parameters of the optimizer
λ = 0.01  # Learning rate
β1 = 0.9  # First moment decay rate
β2 = 0.999  # Second moment decay rate
ϵ = 1e-8  # Epsilon value

# Define the initial position and moment estimates
m = zeros(testfn.dim)
v = zeros(testfn.dim)

ϵsgd = 1e-12
grad_tol = 1e-5

# Setup data structures for measuring GAP
minis = [minimum(sur.y)]
fbest = testfn.f(testfn.xopt...)

# Perform BO loop
∇αxs = []
batch = []

final_locations = []

println("Beginning Bayesian Optimization Loop")
for b in 1:BUDGET
    # Generate a batch of evaluation locations and filter out locations that are close
    # to know sample locations
    batch = generate_batch(BATCH_SIZE; lbs=lbs, ubs=ubs)
    # batch = convert(Matrix{Float64}, filter(x -> !(x in sur.X), batch)')
    
    batch_evals = []
    final_locations = []
    
    # This should be a parallel for loop
    println("---------- BO Iteration #$b ----------")
    bndx = 1
    for x0 in eachcol(batch)
        try
            x0 = convert(Vector{Float64}, x0)

            αxs, ∇αxs = [], []
            ∇αxs = [0., 1., 2.]

            print("\n(Batch #$bndx - $x0) Gradient Ascent Iteration Count: ")
            # Run SGD until convergence
            fprev, fnow = 0., 1.
            for epoch in 1:MAX_SGD_ITERS
                if mod(epoch, 25) == 0 print("|") end
                μx, ∇μx = simulate_trajectory(
                    sur; mc_iters=MC_SAMPLES, rnstream=lds_rns, lbs=lbs, ubs=ubs, x0=x0, h=HORIZON
                )

                # Update gradient vector
                push!(αxs, μx)
                push!(∇αxs, first(∇μx))

                fprev = fnow
                fnow = μx

                # Update x0 based on gradient computation
                # x0, m, v = update_x_adam(x0; ∇g=-∇μx, λ=λ, β1=β1, β2=β2, ϵ=ϵ, m=m, v=v, lbs=lbs, ubs=ubs)
                x0 = update_x(x0; λ=λ, ∇g=∇μx, lbs=lbs, ubs=ubs)

                if abs(fnow - fprev) < ϵsgd || norm(∇μx) < grad_tol
                    println("\nConverged after $epoch epochs")
                    # println("abs(fnow - fprev): $(abs(fnow - fprev)) - fnow: $fnow - fprev: $fprev")
                    break
                end

            end

            push!(batch_evals, αxs[end])
            push!(final_locations, x0)
            bndx += 1
        catch e
            bndx += 1
            println(e)
        end
    end
    # Iterate over batch for best response and sample original process afterwards
    if length(batch_evals) > 0
        println()
        [println("α($(pair[1])) = $(pair[2])") for pair in zip(final_locations, batch_evals)]
        ndx = argmax(batch_evals)
        xnew = final_locations[ndx]

        # Sample original process at x0
        println("\nFinal xnew: $xnew")
        println("--------------------------------------\n")
        res = optimize_hypers_optim(sur, kernel_matern52)
        σ, ℓ = Optim.minimizer(res)
        ψ = kernel_scale(kernel_matern52, [σ, ℓ]);
        sur = fit_surrogate(
            ψ,
            hcat(sur.X, xnew),
            vcat(sur.y, testfn.f(xnew))
        )
        
        push!(minis, minimum(sur.y))
    end
end
