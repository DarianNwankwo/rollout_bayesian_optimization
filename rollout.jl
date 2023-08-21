using Distributed
using Plots
using Sobol
using Distributions 
using LinearAlgebra
using Optim
using ForwardDiff

# addprocs()
# if nworkers() < 8
#     addprocs(8 - nworkers())
# end

# Rename to rollout once refactor is complete
include("lazy_struct.jl")
include("low_discrepancy.jl")
include("optim.jl")
include("radial_basis_surrogates.jl")
include("radial_basis_functions.jl")
include("rbf_optim.jl")
include("trajectory.jl")
include("utils.jl")

function rollout!(T::Trajectory, lbs::Vector{Float64}, ubs::Vector{Float64}; σn2=1e-6, objective=:EI, num_starts=64,
    rnstream, xstarts)
    f0, ∇f0 = gp_draw(T.mfs, T.x0; stdnormal=rnstream[:,1])
    f0, ∇f0 = f0 + T.fs.ymean, ∇f0 .+ T.mfs.∇ymean

    # Evaluate the surrogate at the initial location
    sx0 = T.fs(T.x0)
    # T.opt_HEI = sx0.HEI
    δx0 = rand(length(T.x0))

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_fsurrogate!(T.fs, T.x0, f0)
    update_δsurrogate!(T.δfs, T.fs, δx0, ∇f0)
    update_multioutput_fsurrogate!(T.mfs, T.x0, f0,  ∇f0)

    # Setup up evaluation locations for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.mfs, xnext; stdnormal=rnstream[:, j+1])
        fi, ∇fi = fi + T.fs.ymean, ∇fi .+ T.mfs.∇ymean

        # Evaluate the surrogate at the initial location
        sxi = T.fs(xnext)
        δxi = zeros(length(xnext))
        if sxi.EI > 0
            δxi = -sxi.HEI \ T.δfs(sxi).∇EI
        end

        # Update hessian if a new best is found on trajectory
        # Should fi be fi + T.fs.ymean?
        if fi < T.fmin
            T.fmin = fi
            T.opt_HEI = sxi.HEI
        end
       
        # Update surrogate, perturbed surrogate, and multioutput surrogate
        update_fsurrogate!(T.fs, xnext, fi)
        update_δsurrogate!(T.δfs, T.fs, δxi, ∇fi)
        update_multioutput_fsurrogate!(T.mfs, xnext, fi,  ∇fi)
    end

    return nothing
end

function sample(T::Trajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Cannot sample from a trajectory that has not been rolled out"
    fantasy_slice = T.fs.known_observed + 1 : T.fs.known_observed + T.fs.fantasies_observed
    M = T.fs.known_observed
    return [
        (
            x=T.mfs.X[:,i],
            y=T.mfs.y[i] .+ T.mfs.ymean,
            ∇y=T.mfs.∇y[:,i-M] .+ T.mfs.∇ymean,
        ) for i in fantasy_slice
    ]
end

function best(T::Trajectory)
    # step[2] corresponds to the function value
    path = sample(T)
    _, minndx = findmin([step.y for step in path])
    return minndx, path[minndx]
end

function α(T::Trajectory)
    m = T.fs.known_observed - 1
    path = sample(T)
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    fb = best_step.y
    return max(fmini - fb, 0.)
end

function ∇α(T::Trajectory)
    if T.h == 0 return T.fs(T.x0).∇EI end
    m = T.fs.known_observed - 1
    fmini = minimum(T.s.y) + T.s.ymean
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step
    if fmini <= fb
        return zeros(length(xb))
    end
    # Investigate the computations from the ground up and 
    # see where the sign error might have been introduced
    return transpose(-∇fb'*T.opt_HEI)
end


function visualize1D(T::Trajectory)
    p = plot(
        0:T.h, T.xfs[1, :], color=:red, label=nothing, xlabel="Decision Epochs (h=$(T.h))",
        ylabel="Control Space (xʳ)", title="Trajectory Visualization in 1D",
        xticks=(0:T.h, ["x$(i)" for i in 0:T.h]), xrotation=45, grid=false
    )
    vline!(0:T.h, color=:black, linestyle=:dash, linewidth=1, label=nothing, alpha=.2)
    scatter!(0:T.h, T.xfs[1, :], color=:red, label=nothing)
    # lbs, ubs not defined
    yticks!(
        round.(range(lbs[1], ubs[1], length=11), digits=1)
    )

    best_ndx, best_step = best(T)
    scatter!([best_ndx-1], [best_step.x[1]], color=:green, label="Best Point")

    return p
end


function simulate_trajectory(sur::RBFsurrogate; objective=:EI, x0, mc_iters, rnstream, lbs, ubs, h)
    sx = sur(x0)
    αxi, ∇αxi = 0., zeros(size(sur.X, 1))

    for sample in 1:mc_iters
        fsur = Base.deepcopy(sur)
        fantasy_ndx = size(fsur.X, 2) + 1

        # Rollout trajectory
        T = Trajectory(fsur, x0, fantasy_ndx; h=h, fopt=minimum(sur.y) + sur.ymean)
        rollout!(T, lbs, ubs; rnstream=rnstream[sample,:,:], objective=objective)

        # Evaluate rolled out trajectory
        αxi += α(T)
        ∇αxi .+= ∇α(T)
    end

    # Average trajectories MC simulation
    μx = αxi / mc_iters
    ∇μx = ∇αxi / mc_iters

    return μx + sx.EI, ∇μx
end