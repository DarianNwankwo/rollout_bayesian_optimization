using Plots
@everywhere using Sobol

# Rename to rollout once refactor is complete
include("lazy_struct.jl")
include("low_discrepancy.jl")
include("optim.jl")
include("radial_basis_surrogates_update.jl")
include("radial_basis_functions.jl")
include("rbf_optim.jl")
include("trajectory.jl")
include("utils.jl")

"""
TODO: We should be able to use the same trajectory object to compute
trajectory samples instead of reconstructing a new object when the location
hasn't changed.
"""

@everywhere function rollout!(T::Trajectory, lbs::Vector{Float64}, ubs::Vector{Float64}; σn2=1e-6,
    rnstream)
    fbest = T.fopt
    x0 = T.xfs[:, 1]
    
    # (Inquiry) This assumption about having gradient information might be hurting us
    # Sample a fantasized value and gradient at x0
    f0 = gp_draw(T.s, x0, stdnormal=rnstream[1,1])
    h, E, ∇f0 = 1e-8, I(length(x0)), zeros(length(x0))
    for d in 1:length(x0)
        xplus = x0 + E[:, d] * h
        xminus = x0 - E[:, d] * h
        fplus = gp_draw(T.s, xplus, stdnormal=rnstream[d, 1])
        fminus = gp_draw(T.s, xminus, stdnormal=rnstream[d, 1])
        ∇f0[d] = (fplus - fminus) / (2h)
    end
    sx0 = T.s(x0)
    T.opt_HEI = sx0.HEI
    δsx0 = -sx0.HEI \ T.δs(sx0).∇EI
    # δsx0 = zeros(length(x0))

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    T.s = update_surrogate(T.s, x0, f0)
    T.δs = update_δsurrogate(T.s, T.δs, δsx0, ∇f0)
    T.ms = update_multioutput_surrogate(T.ms, x0, f0, ∇f0)

    T.ys[1] = f0
    T.∇ys[:, 1] = ∇f0
    T.xfs[:, 1] = x0

    # Perform rollout for the fantasized trajectories
    ϵ = 1e-6
    s = SobolSeq(lbs, ubs)
    xstarts = reduce(hcat, next!(s) for i = 1:64)
    xstarts = hcat(xstarts, lbs .+ ϵ)
    xstarts = hcat(xstarts, ubs .- ϵ)
    # xstarts = randsample(10, length(∇f0), lbs, ubs) # Probably should parametrize this number
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.s, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.ms, xnext; stdnormal=rnstream[:, j+1])
        
        # Compute variations in xnext and update surrogate, perturbed surrogate,
        # and multioutput surrogate
        sxnext = T.s(xnext)
        # println("(h=$j)xnext: $xnext - sxnext.HEI: $(sxnext.HEI) - sxnext.EI: $(sxnext.EI)")
        # δsxnext = -sxnext.HEI \ T.δs(sxnext, T.fantasy_ndx).∇EI
        δsxnext = zeros(length(xnext))
        if sxnext.EI > 0
            δsxnext = -sxnext.HEI \ T.δs(sxnext).∇EI
        end
        # δsxnext = -sxnext.HEI \ T.δs(sxnext).∇EI

        # Update hessian if a new best is found on trajectory
        if fi < fbest
            fbest = fi
            T.opt_HEI = sxnext.HEI
        end
        
        T.s = update_surrogate(T.s, xnext, fi)
        T.δs = update_δsurrogate(T.s, T.δs, δsxnext, ∇fi)
        T.ms = update_multioutput_surrogate(T.ms, xnext, fi, ∇fi)

        # Update trajectory path
        T.xfs[:, j+1] = xnext
        T.ys[j+1] = fi
        T.∇ys[:, j+1] = ∇fi
    end
end

function sample(T::Trajectory)
    path = [(x=T.xfs[:,i], y=T.ys[i], ∇y=T.∇ys[:,i]) for i in 1:T.h+1]
    return path
end

function best(T::Trajectory)
    # step[2] corresponds to the function value
    path = sample(T)
    _, minndx = findmin([step.y for step in path])
    return minndx, path[minndx]
end

function α(T::Trajectory)
    m = T.fantasy_ndx-1
    path = sample(T)
    fmini = T.fopt
    best_ndx, best_step = best(T)
    fb = T.ys[best_ndx]
    return max(fmini - fb, 0.)
end

function ∇α(T::Trajectory)
    m = T.fantasy_ndx-1
    fmini = T.fopt
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step
    if fmini <= fb
        return zeros(length(xb))
    end
    # Investigate the computations from the ground up and 
    # see where the sign error might have been introduced
    return transpose(∇fb'*T.opt_HEI)
end


function visualize1D(T::Trajectory)
    p = plot(
        0:T.h, T.xfs[1, :], color=:red, label=nothing, xlabel="Decision Epochs (h=$(T.h))",
        ylabel="Control Space (xʳ)", title="Trajectory Visualization in 1D",
        xticks=(0:T.h, ["x$(i)" for i in 0:T.h]), xrotation=45, grid=false
    )
    vline!(0:T.h, color=:black, linestyle=:dash, linewidth=1, label=nothing, alpha=.2)
    scatter!(0:T.h, T.xfs[1, :], color=:red, label=nothing)
    yticks!(
        round.(range(lbs[1], ubs[1], length=11), digits=1)
    )

    best_ndx, best_step = best(T)
    scatter!([best_ndx-1], [best_step.x[1]], color=:green, label="Best Point")

    return p
end


function simulate_trajectory(sur::RBFsurrogate; x0, mc_iters, rnstream, lbs, ubs, h)
    αxi, ∇αxi = 0., zeros(size(sur.X, 1))

    for sample in 1:mc_iters
        fsur = Base.deepcopy(sur)
        fantasy_ndx = size(fsur.X, 2) + 1

        # Rollout trajectory
        T = Trajectory(fsur, x0, fantasy_ndx; h=h, fopt=minimum(sur.y))
        rollout!(T, lbs, ubs; rnstream=rnstream[sample,:,:])

        # Evaluate rolled out trajectory
        αxi += α(T)
        ∇αxi .+= ∇α(T)
    end

    # Average trajectories MC simulation
    μx = αxi / mc_iters
    ∇μx = ∇αxi / mc_iters

    return μx, ∇μx
end