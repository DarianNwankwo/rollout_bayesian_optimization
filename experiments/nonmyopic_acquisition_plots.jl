if length(ARGS) < 2
    println("Usage: julia nonmyopic_acquisition_plots.jl <HORIZON> <MC_SAMPLES>")
    exit(1)
end

using Distributions
using LinearAlgebra
using Plots
using Test
using Sobol
using Optim
using Random

include("../rollout.jl")
include("../testfns.jl")


function ei(μ, σ, fbest)
    z = (fbest - μ) / σ
    Φz = Distributions.normcdf(z)
    ϕz = Distributions.normpdf(z)
    return σ*(z*Φz + ϕz)
end

function poi(μ, σ, fbest)
    z = (fbest - μ) / σ
    Φz = Distributions.normcdf(z)
    return Φz
end

# Global parameters
HORIZON = parse(Int64, ARGS[1])
MC_SAMPLES = parse(Int64, ARGS[2])
# USE_LOW_DISCREPANCY = parse(Int64, ARGS[3]) == 1 ? true : false

# Setup toy problem
testfn = TestFunction(
    1, [0. 1.], [.5],
    x -> 0.,
    ∇x -> [0.]
)
lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]

# Setup low discrepancy random number stream
lds_rns = gen_low_discrepancy_sequence(MC_SAMPLES, testfn.dim, HORIZON+1)
rns = randn(MC_SAMPLES, testfn.dim+1, HORIZON+1)

# Gather initial samples/experimental data
N, θ = 1, [.25]
X = [.15, .85]
X = reshape(X, 1, length(X))
ψ = kernel_scale(kernel_matern52, [1., θ...])
sur = fit_surrogate(ψ, X, testfn.f)

domain = filter(x -> !(x in X), lbs[1]:.01:ubs[1])

eis, ∇eis = [], []
for random_number_stream in [lds_rns, rns]
    rollout_ei = [0 0] # Sample mean and variance tuples
    ∇rollout_ei = [0 0] # Sample mean and variance tuples
    
    println("Total '|'s => $(length(domain))")
    # Iterate over each input location
    for x0 in domain
    # for x0 in [.5]
        print("|$x0")
        # Grab each input location and convert to a column vector
        x0 = [x0]

        αxs, ∇αxs = [], []
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
            push!(αxs, α(T))
            push!(∇αxs, first(∇α(T)))
        end # endfor sample

        # Average trajectories
        μx = sum(αxs) / length(αxs)
        ∇μx = sum(∇αxs) / length(αxs)
        σx = sum((αxs .- μx) .^ 2) / (MC_SAMPLES-1)
        ∇σx = sum((∇αxs .- ∇μx) .^ 2) / (MC_SAMPLES-1)

        # Update history
        sx = sur(x0)
        μx += ei(sx.μ, sx.σ, minimum(sur.y)) + poi(sx.μ, sx.σ, minimum(sur.y))
        rollout_ei = vcat(rollout_ei, [μx σx])
        ∇rollout_ei = vcat(∇rollout_ei, [∇μx ∇σx])
    end # endfor x0
    rollout_ei = rollout_ei[2:end, :]
    ∇rollout_ei = ∇rollout_ei[2:end, :];
    
    push!(eis, rollout_ei)
    push!(∇eis, ∇rollout_ei)
end


dir = dirname(@__FILE__)
plot_domain = range(lbs[1], ubs[1], length=length(eis[2][:, 2]))
# ylims = (minimum(eis[2][:, 1]) - 2*maximum(eis[2][:, 2]),
#     maximum(eis[2][:, 1]) + 2*maximum(eis[2][:, 2]))
# ∇ylims = (minimum(∇eis[2][:, 1]) - 2*maximum(∇eis[2][:, 1]),
#     maximum(∇eis[2][:, 1]) + 2*maximum(∇eis[2][:, 1]))
# ∇ylims_lds = (minimum(∇eis[1][:, 1]) - 2*maximum(∇eis[1][:, 1]),
#     maximum(∇eis[1][:, 1]) + 2*maximum(∇eis[1][:, 1]))

# Save individual plot for non-low discrepancy sequence simulation
plot(plot_domain, eis[2][:, 1], ribbons=sqrt.(eis[2][:, 2]),
    label="EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
savefig("$(dir)/plots/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES).png")

# Save individual plot for low discrepancy sequence simulation
plot(plot_domain, eis[1][:, 1], ribbons=sqrt.(eis[1][:, 2]),
    label="LDS EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
savefig("$(dir)/plots/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES)_lds.png")

# Save plot with graphs stacked for comparison
plot(plot_domain, eis[2][:, 1], ribbons=sqrt.(eis[2][:, 2]),
    label="EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
plot!(plot_domain, eis[1][:, 1], ribbons=sqrt.(eis[1][:, 2]),
    label="LDS EI(h=$HORIZON)", linestyle=:dash
)
savefig("$(dir)/plots/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked.png")

# Save similar plots for gradients
plot(plot_domain, ∇eis[2][:, 1], ribbons=sqrt.(∇eis[2][:, 2]),
    label="EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims
)
savefig("$(dir)/plots/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES).png")

plot(plot_domain, ∇eis[1][:, 1], ribbons=sqrt.(∇eis[1][:, 2]),
    label="LDS EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims_lds
)
savefig("$(dir)/plots/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES)_lds.png")

plot(plot_domain, ∇eis[2][:, 1], ribbons=sqrt.(∇eis[2][:, 2]),
    label="EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims
)
plot!(plot_domain, ∇eis[1][:, 1], ribbons=sqrt.(∇eis[1][:, 2]),
    label="LDS EI(h=$HORIZON)", linestyle=:dash
)
savefig("$(dir)/plots/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked.png")