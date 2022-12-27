# Rename to rollout once refactor is complete
include("lazy_struct.jl")
include("low_discrepancy.jl")
include("optim.jl")
include("radial_basis_surrogates.jl")
include("radial_basis_functions.jl")
include("rbf_optim.jl")
include("trajectory.jl")
include("utils.jl")

"""
TODO: We should be able to use the same trajectory object to compute
trajectory samples instead of reconstructing a new object when the location
hasn't changed.
"""
function rollout!(T::Trajectory, lbs::Vector{Float64}, ubs::Vector{Float64};
    rnstream)
    fbest = minimum(T.s.y)
    x0 = T.xfs[:, 1]
    
    # (Inquiry) This assumption about having gradient information might be hurting us
    # Sample a fantasized value and gradient at x0
    # f0, ∇f0 = dgp_draw(T.s, x0, dim=length(x0), stdnormal=rnstream[:,1])
    f0 = gp_draw(T.s, x0, stdnormal=rnstream[1,1])
    f0plush = gp_draw(T.s, x0 + [1e-8], stdnormal=rnstream[1,1])
    ∇f0 = [(f0plush - f0) / 1e-8]
    sx0 = T.s(x0)
    T.opt_HEI = sx0.HEI
    # δsx0 = -sx0.HEI \ T.δs(sx0, T.fantasy_ndx).∇EI
    δsx0 = -sx0.HEI \ T.δs(sx0).∇EI

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    T.s = update_surrogate(T.s, x0, f0)
    T.δs = update_δsurrogate(T.s, T.δs, δsx0, ∇f0)
    T.ms = update_multioutput_surrogate(T.ms, x0, f0, ∇f0)

    T.ys[1] = f0
    T.∇ys[:, 1] = ∇f0
    T.xfs[:, 1] = x0

    # Perform rollout for the fantasized trajectories
    xstarts = randsample(10, length(∇f0), lbs, ubs) # Probably should parametrize this number
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.s, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        # fi, ∇fi = gp_draw(T.ms, xnext; stdnormal=rnstream[:, j+1])
        fi = gp_draw(T.s, xnext, stdnormal=rnstream[1,1])
        fiplush = gp_draw(T.s, xnext + [1e-8], stdnormal=rnstream[1,1])
        ∇fi = [(fiplush - fi) / 1e-8]
        
        # Compute variations in xnext and update surrogate, perturbed surrogate,
        # and multioutput surrogate
        sxnext = T.s(xnext)
        δsxnext = -sxnext.HEI \ T.δs(sxnext, T.fantasy_ndx).∇EI
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
    path = [(T.xfs[:,i], T.ys[i], T.∇ys[:,i]) for i in 1:T.h+1]
    return path
end

function best(T::Trajectory)
    # step[2] corresponds to the function value
    path = sample(T)
    _, minndx = findmin([step[2] for step in path])
    return minndx, path[minndx]
end

function α(T::Trajectory)
    m = T.fantasy_ndx-1
    path = sample(T)
    fmini = minimum(T.s.y[1:m])
    best_ndx, best_step = best(T)
    fb = T.ys[best_ndx]
    return max(fmini - fb, 0.)
end

function ∇α(T::Trajectory)
    m = T.fantasy_ndx-1
    fmini = minimum(T.s.y[1:m])
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step
    if fmini <= fb
        return zeros(length(xb))
    end
    return transpose(-∇fb'*T.opt_HEI)
end