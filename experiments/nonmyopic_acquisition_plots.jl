using Distributions
using LinearAlgebra
using Plots
using Test
using Sobol
using Optim
using Random

include("../rollout.jl")
include("../testfns.jl")

if length(ARGS) < 2
    println("Usage: julia nonmyopic_acquisition_plots.jl <HORIZON> <MC_SAMPLES>")
    exit(1)
end

# Global parameters
HORIZON = parse(Int64, ARGS[1])
MC_SAMPLES = parse(Int64, ARGS[2])

# Setup toy problem
testfn = TestFunction(
    1, [0. 1.], [.5],
    x -> 0.,
    ∇x -> [0.]
)
lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

# Setup low discrepancy random number stream
lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON+1);
rns = randn(MC_SAMPLES, testfn.dim+1, HORIZON+1);

# Gather initial samples/experimental data
N, θ = 1, [.25]
X = [.15, .85]
X = reshape(X, 1, length(X))
ψ = kernel_scale(kernel_matern52, [5., θ...])
sur = fit_surrogate(ψ, X, testfn.f)

domain = filter(x -> !(x in X), lbs[1]:.01:ubs[1])

rollout_ei = []
∇rollout_ei = []

for x0 in domain
    # Grab each input location and convert to a column vector
    x0 = [x0]
    # sx = sur(x0)
    # cv_ei = sx.EI
    # μ, σ, fb = sx.μ, sx.σ, minimum(sur.y)
    # cv_poi = 1 - Distributions.normcdf((fb-μ)/σ)

    αx, ∇αx = 0., [0.]
    # Monte-carlo integrate trajectory for x0
    for sample in 1:MC_SAMPLES
        # Make a copy of our surrogate to pass to the trajectory struct
        # for fantasized computations
        fsur = Base.deepcopy(sur)
        fantasy_ndx = size(fsur.X, 2) + 1

        # Rollout trajectory
        T = Trajectory(fsur, x0, fantasy_ndx; h=HORIZON)
        rollout!(T, lbs, ubs; rnstream=random_number_stream[sample,:,:])

        # Evaluate rolled out trajectory
        αx += α(T)
        ∇αx += ∇α(T)
    end # endfor sample

    # Average trajectories
    αx /= MC_SAMPLES
    ∇αx /= MC_SAMPLES
    
    # Add control variates
    # αx += cv_ei
    # αx += cv_poi
    # ∇αx += [cv_ei]
    # ∇αx += [cv_poi]

    # Update history
    push!(rollout_ei, αx)
    push!(∇rollout_ei, first(∇αx))
end # endfor x0


plot(
    dense_domain,
    rollout_ei,
    label="Rollout EI(h=$HORIZON)",
    linestyle=:dash
)