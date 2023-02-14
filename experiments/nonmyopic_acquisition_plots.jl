if length(ARGS) < 2
    println("Usage: julia nonmyopic_acquisition_plots.jl <HORIZON> <MC_SAMPLES>")
    exit(1)
end

using Measures
using Dates
using Distributions
using LinearAlgebra
using Plots
using Test
using Sobol
using Optim
using Random
using Base.Filesystem

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
σn2 = 1e-6
X = [.05, .95]
X = reshape(X, 1, length(X))
ψ = kernel_scale(kernel_matern52, [1., θ...])
sur = fit_surrogate(ψ, X, testfn.f; σn2=σn2)

domain = filter(x -> !(x in X), lbs[1]:.01:ubs[1])

eis, ∇eis = [], []
for (ndx, random_number_stream) in enumerate([lds_rns, rns])
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
            T = Trajectory(fsur, x0, fantasy_ndx; h=HORIZON, fopt=minimum(sur.y))
            rollout!(T, lbs, ubs; rnstream=random_number_stream[sample,:,:])

            # Evaluate rolled out trajectory
            push!(αxs, α(T))
            push!(∇αxs, first(∇α(T)))
        end # endfor sample

        # Average trajectories
        μx = sum(αxs) / MC_SAMPLES
        ∇μx = sum(∇αxs) / MC_SAMPLES
        σx = sum((αxs .- μx) .^ 2) / (MC_SAMPLES-1)
        ∇σx = sum((∇αxs .- ∇μx) .^ 2) / (MC_SAMPLES-1)

        # Update history
        if ndx == 1
            # Add control variates to QMC estimates
            sx = sur(x0)
            μx += ei(sx.μ, sx.σ, minimum(sur.y)) + poi(sx.μ, sx.σ, minimum(sur.y))
        end
        rollout_ei = vcat(rollout_ei, [μx σx])
        ∇rollout_ei = vcat(∇rollout_ei, [∇μx ∇σx])
    end # endfor x0
    rollout_ei = rollout_ei[2:end, :]
    ∇rollout_ei = ∇rollout_ei[2:end, :];
    
    push!(eis, rollout_ei)
    push!(∇eis, ∇rollout_ei)
end



filename, extension = splitext(basename(@__FILE__))
dir_name = "plots/" * filename * "/contrived_h$(HORIZON)"
mkpath(dir_name)
plot_domain = range(lbs[1], ubs[1], length=length(eis[2][:, 2]))

# Save individual plot for non-low discrepancy sequence simulation
plot(plot_domain, eis[2][:, 1], ribbons=sqrt.(eis[2][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="MC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES).png")

# Save individual plot for low discrepancy sequence simulation
plot(plot_domain, eis[1][:, 1], ribbons=sqrt.(eis[1][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="QMC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_qmc$(MC_SAMPLES).png")

# Save plot with graphs stacked for comparison
plot(plot_domain, eis[2][:, 1], ribbons=sqrt.(eis[2][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="MC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=ylims
)
plot!(plot_domain, eis[1][:, 1], ribbons=sqrt.(eis[1][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="QMC LDS EI(h=$HORIZON)±σ", linestyle=:dash
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked.png")

# Save similar plots for gradients
plot(plot_domain, ∇eis[2][:, 1], ribbons=sqrt.(∇eis[2][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="MC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=∇ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES).png")

plot(plot_domain, ∇eis[1][:, 1], ribbons=sqrt.(∇eis[1][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="QMC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=∇ylims_lds
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_qmc$(MC_SAMPLES).png")

plot(plot_domain, ∇eis[2][:, 1], ribbons=sqrt.(∇eis[2][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="MC EI(h=$HORIZON)±σ", linestyle=:dash#, ylims=∇ylims
)
plot!(plot_domain, ∇eis[1][:, 1], ribbons=sqrt.(∇eis[1][:, 2]) / sqrt(MC_SAMPLES), margin=10mm,
    label="QMC EI(h=$HORIZON)±σ", linestyle=:dash
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked.png")

#######################################################
## Save all the same plots without the error ribbons ##
#######################################################
# Save individual plot for non-low discrepancy sequence simulation
plot(plot_domain, eis[2][:, 1], margin=10mm,
    label="MC EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES)_no_ribbons.png")

# Save individual plot for low discrepancy sequence simulation
plot(plot_domain, eis[1][:, 1], margin=10mm,
    label="QMC EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_qmc$(MC_SAMPLES)_no_ribbons.png")

# Save plot with graphs stacked for comparison
plot(plot_domain, eis[2][:, 1], margin=10mm,
    label="MC EI(h=$HORIZON)", linestyle=:dash#, ylims=ylims
)
plot!(plot_domain, eis[1][:, 1], margin=10mm,
    label="QMC EI(h=$HORIZON)", linestyle=:dash
)
# vline!(X)
savefig("$(dir_name)/rollout_ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked_no_ribbons.png")

# Save similar plots for gradients
plot(plot_domain, ∇eis[2][:, 1], margin=10mm,
    label="MC EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES)_no_ribbons.png")

plot(plot_domain, ∇eis[1][:, 1], margin=10mm,
    label="QMC EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims_lds
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_qmc$(MC_SAMPLES)_no_ribbons.png")

plot(plot_domain, ∇eis[2][:, 1], margin=10mm,
    label="MC EI(h=$HORIZON)", linestyle=:dash#, ylims=∇ylims
)
plot!(plot_domain, ∇eis[1][:, 1], margin=10mm,
    label="QMC EI(h=$HORIZON)", linestyle=:dash
)
# vline!(X)
savefig("$(dir_name)/rollout_∇ei_h$(HORIZON)_mc$(MC_SAMPLES)_stacked_no_ribbons.png")