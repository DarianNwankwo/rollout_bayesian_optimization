# Parse command line arguments
cli_args = ["HORIZON", "MC_SAMPLES", "BUDGET", "NUM_TRIALS", "TESTFUNC_NAME", "TESTFUNC_ARGS"]
if length(ARGS) < length(cli_args) - 1
    local str_builder = "Usage: julia bayesopt1d.jl "
    for arg in cli_args
        str_builder *= "<$arg> "
    end
    println(str_builder)
    exit(1)
end

using Measures
using Distributed
using Distributions
using LinearAlgebra
using Plots
using SharedArrays
using Sobol
using Random

addprocs()
println("Total Workers: $(nworkers())")

@everywhere include("../rollout.jl")
@everywhere include("../testfns.jl")
@everywhere include("./utils.jl")

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
BATCH_SIZE = 64
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
θ, output_var, σn2 = [1.], 1., 1e-4
ψ = kernel_scale(kernel_matern52, [output_var, θ...]);

# Generate batch of locations to perform SGA on
batch = generate_batch(BATCH_SIZE; lbs=lbs, ubs=ubs)

for trial in 1:NUM_TRIALS
    global batch_results = []
    # Setup structures for collection trial data
    X = reshape(initial_samples[:, trial], testfn.dim, 1)
    sur = fit_surrogate(ψ, X, testfn.f; σn2=σn2) # (TODO): Learn kernel hyperparameters

    filename, extension = splitext(basename(@__FILE__))
    dirs = [func_name]
    dir_name = create_experiment_directory(filename, dirs)

    domain = filter(x -> !(x in sur.X), lbs[1]:.01:ubs[1])

    println("Beginning Bayesian Optimization Main Loop")
    println("-----------------------------------------")
    for budget in 1:BUDGET
        plot1DEI(sur; domain=domain)
        plot!(margin=10mm)
        savefig("$dir_name/$(filename)_$(func_name)_h$(HORIZON)_$(MC_SAMPLES)_budget$(BUDGET)_trial$(trial)_ei.png")
        println("Iteration #$budget")
        # Optimize each batch location in parallel
        results = []

        results = @distributed (append!) for j = 1:size(batch, 2)
            # Setup parameters for gradient ascent for each process
            xbegin, xend = batch[:, j], batch[:, j]
            vt, γ, η = zeros(testfn.dim), .9, .01
            ϵgrad, ϵfunc, num_increases_count = 1e-8, 1e-8, 0
            num_increases = 3
            αxs, ∇αxs = [], []
            iters = 1

            # Main loop for Stochastic Gradient Ascent
            # TODO: Figure out why gradient estimates are very negative at 1.1
            try
                for epoch in 1:MAX_SGD_ITERS
                    # Compute stochastic estimates of acquisition function and gradient
                    μx, ∇μx = 0., zeros(testfn.dim)
                    x0 = xend + γ*vt
                    x0 = max.(x0, lbs)
                    x0 = min.(x0, ubs)
                    μx, ∇μx = simulate_trajectory(
                        sur; mc_iters=MC_SAMPLES, rnstream=lds_rns, lbs=lbs, ubs=ubs, x0=x0, h=HORIZON
                    )
                    # Perform update rule for nesterov accelerated gradient
                    vt = γ*vt + η*∇μx
                    xend = xend + vt
                    xend = max.(xend, lbs)
                    xend = min.(xend, ubs)

                    # Cache values
                    if xbegin == [1.1]
                        println("$epoch.) μx: $μx - ∇μx: $∇μx - xend: $xend -- x0: $x0")
                    end
                    push!(αxs, μx)
                    push!(∇αxs, ∇μx)

                    # Check for convergence
                    iters = epoch
                    if epoch > 1 && abs(αxs[end] - αxs[end-1]) < ϵfunc
                        num_increases_count += 1
                        if num_increases_count >= num_increases
                            break
                        end
                    end
                end # END for epoch in 1:MAX_SGD_ITERS

                # println("Finished SGD for xbegin. length(αxs): $(length(αxs))")
                if length(αxs) > 0
                    # push!(results, (start=xbegin, finish=xend, func=αxs[end], grad=∇αxs[end], iters=iters))
                    [(start=xbegin, finish=xend, func=αxs[end], grad=∇αxs[end], iters=iters)]
                end  
                catch err
                    println("Error: $err")
                    println("-----------------------------------------")
                    println("Initial location x0=$xbegin is too close to some point in X=$(sur.X)\n")
                    [(start=xbegin, finish=xend, func=-Inf, grad=[-Inf], iters=iters)]
                end
        end # END @distributed
        
        global batch_results = shuffle(results)
        for (ndx, res) in enumerate(batch_results)
            println("#$ndx: start=$(res.start), finish=$(res.finish), func=$(res.func), grad=$(res.grad), iters=$(res.iters)")
        end

        # Update surrogate with element that optimize the acquisition function
        max_ndx = findmax(t -> t.func, batch_results)[2]
        max_pairing = batch_results[max_ndx]
        xnew = max_pairing.finish
        recover_y = sur.y .+ sur.ymean

        sur = fit_surrogate(
            ψ,
            hcat(sur.X, xnew),
            vcat(recover_y, testfn.f(xnew));
            σn2=σn2
        )
        println("Updated X: $(sur.X)\n")
        res = optimize_hypers_optim(sur, kernel_matern52; σn2=σn2)
        σ, ℓ = Optim.minimizer(res)
        global ψ = kernel_scale(kernel_matern52, [σ, ℓ])

        recover_y = sur.y .+ sur.ymean
        sur = fit_surrogate(ψ, sur.X, recover_y; σn2=σn2)
    end # END Bayesian Optimization Loop
    println("-----------------------------------------")
    # Update collective data

    # filename, extension = splitext(basename(@__FILE__))
    # dirs = [func_name]
    # dir_name = create_experiment_directory(filename, dirs)

    # domain = filter(x -> !(x in sur.X), lbs[1]:.01:ubs[1])
    plot1D(sur; domain=domain)
    plot!(margin=10mm)
    savefig("$dir_name/$(filename)_$(func_name)_h$(HORIZON)_$(MC_SAMPLES)_budget$(BUDGET)_trial$(trial).png")

    # plot1DEI(sur; domain=domain)
    # plot!(margin=10mm)
    # savefig("$dir_name/$(filename)_$(func_name)_h$(HORIZON)_$(MC_SAMPLES)_budget$(BUDGET)_trial$(trial)_ei.png")

    println("Saving results to $dir_name")
end


# filename, extension = splitext(basename(@__FILE__))
# dirs = [func_name]
# dir_name = create_experiment_directory(filename, dirs)

# domain = filter(x -> !(x in sur.X), lbs[1]:.01:ubs[1])
# plot1D(sur; domain=domain)
# savefig("$dir_name/$(filename)_$(func_name)_$(func_args)_$(HORIZON)_$(MC_SAMPLES)_$(BUDGET)_$(NUM_TRIALS).png")

# plot1DEI(sur; domain=domain)
# savefig("$dir_name/$(filename)_$(func_name)_$(func_args)_$(HORIZON)_$(MC_SAMPLES)_$(BUDGET)_$(NUM_TRIALS)_ei.png")

# println("Saving results to $dir_name")