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
include("testfns.jl")


function not_near(x::Vector{Float64}, X::Matrix{Float64}; tol::Float64=1e-6)
    return all([norm(x - X[:, i]) > tol for i in 1:size(X, 2)])
end

"""
I THINK MY COVARIANCE MEASURES FOR GRADIENTS IS WRONG

If we happen to be near a known location, we shouldn't sample from our multioutput surrogate, but rather our
ground truth single output surrogate, I suspect. Here is what is going on:

Our first sample on our fantasized trajectory isn't driven by our policy explicitly. It says that if EI told us to sample
at x0, what is the anticipated behavior, after using EI, for the next h samples. So, our trajectory contains h+1 samples,
where the first sample is at x0, and the next h samples are driven by EI. So h+1 fantasized samples, h of which are driven
by EI. When x0 happens to be near a known location, we observe a tendency of the rollout acquisition function to blowup.
I think this occurs when x0 is a known location that is close to the best thing we've seen thus far.

This means that if EI told us to sample at a known location and then we rollout our trajectory and there is some
expected reduction in objective, our rollout acquisition function is going to say x0 is a decent location to sample our
original process at.

When we sample gradient information at x0, when x0 is near a known point, we have a tendency to learn an approximation
of the underlying function that is not accurate.
"""
function rollout!(T::Trajectory, lbs::Vector{Float64}, ubs::Vector{Float64};
    rnstream::Matrix{Float64}, xstarts::Matrix{Float64})
    # Initial draw at predetermined location not chosen by policy
    f0, ∇f0 = gp_draw(T.mfs, T.x0; stdnormal=rnstream[:,1])

    # Evaluate the surrogate at the initial location
    sx0 = T.fs(T.x0)
    δx0 = rand(length(T.x0))

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_trajectory!(T, T.x0, δx0, f0, ∇f0)

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.mfs, xnext; stdnormal=rnstream[:, j+1])

        # Evaluate the surrogate at the suggested location
        sxi = T.fs(xnext)
        δxi = zeros(length(xnext))
        if sxi.EI > 0
            δxi = -sxi.HEI \ T.δfs(sxi).∇EI
            # Update hessian if a new best is found on trajectory
            trajectory_start = T.mfs.known_observed + 1
            if fi < T.fmin && not_near(xnext, T.mfs.X[:, trajectory_start:end])
                T.fmin = fi
                T.opt_HEI = sxi.HEI
            end
        end
       
        # Update surrogate, perturbed surrogate, and multioutput surrogate
        update_trajectory!(T, xnext, δxi, fi,  ∇fi)
    end

    return nothing
end



function rollout_deterministic!(T::Trajectory, lbs::Vector{Float64}, ubs::Vector{Float64};
    xstarts::Matrix{Float64}, testfn::TestFunction)
    f0, ∇f0 = testfn.f(T.x0), testfn.∇f(T.x0)

    # Evaluate the surrogate at the initial location
    sx0 = T.fs(T.x0)
    δx0 = rand(length(T.x0))

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_trajectory!(T, T.x0, δx0, f0, ∇f0)

    # Setup up evaluation locations for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = testfn.f(xnext), testfn.∇f(xnext)

        # Evaluate the surrogate at the suggested location
        sxi = T.fs(xnext)
        δxi = zeros(length(xnext))

        if sxi.EI > 0
            δxi = -sxi.HEI \ T.δfs(sxi).∇EI
        end

        # Update hessian if a new best is found on trajectory
        if fi < T.fmin
            T.fmin = fi
            T.opt_HEI = sxi.HEI
        end
       
        # Update surrogate, perturbed surrogate, and multioutput surrogate
        update_trajectory!(T, xnext, δxi, fi,  ∇fi)
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
    path = sample(T)
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    fb = best_step.y
    return max(fmini - fb, 0.)
end


function ∇α(T::Trajectory)
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step
    if fmini <= fb
        return zeros(length(xb))
    end

    return transpose(-∇fb'*T.opt_HEI)
end

function ∇αd(T::Trajectory)
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step
    if fmini <= fb
        return zeros(length(xb))
    end

    return transpose(-∇fb'*T.opt_HEI)
end


function visualize1D(T::Trajectory)
    known_observed = T.fs.known_observed
    xfs = T.mfs.X[:, known_observed+1:end]
    p = plot(
        0:T.h, xfs[1, :], color=:red, label=nothing, xlabel="Decision Epochs (h=$(T.h))",
        ylabel="Control Space (xʳ)", title="Trajectory Visualization in 1D",
        xticks=(0:T.h, ["Step $(i)" for i in 0:T.h]), xrotation=45, grid=false
    )
    vline!(0:T.h, color=:black, linestyle=:dash, linewidth=1, label=nothing, alpha=.2)
    scatter!(0:T.h, xfs[1, :], color=:red, label=nothing)
    # lbs, ubs not defined
    yticks!(
        round.(range(lbs[1], ubs[1], length=11), digits=1)
    )

    best_ndx, best_step = best(T)
    scatter!([best_ndx-1], [best_step.x[1]], color=:green, label="Best Point")

    return p
end


function simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    variance_reduction::Bool=false
    )
    αxs, ∇αxs = zeros(tp.mc_iters), zeros(length(tp.x0), tp.mc_iters)
    deepcopy_s = Base.deepcopy(s)

    for sample_ndx in 1:tp.mc_iters
        # Rollout trajectory
        T = Trajectory(deepcopy_s, tp.x0, tp.h)
        rollout!(T, tp.lbs, tp.ubs;
            rnstream=tp.rnstream_sequence[sample_ndx, :, :],
            xstarts=xstarts
        )
        
        # Evaluate rolled out trajectory
        αxs[sample_ndx] = α(T)
        ∇αxs[:, sample_ndx] = ∇α(T)
    end

    # Average trajectories
    μx = sum(αxs) / tp.mc_iters
    ∇μx = vec(sum(∇αxs, dims=2) / tp.mc_iters)
    stderr_μx = sqrt(sum((αxs .- μx) .^ 2) / (tp.mc_iters - 1))
    stderr_∇μx = sqrt(sum((∇αxs .- ∇μx) .^ 2) / (tp.mc_iters - 1))

    if variance_reduction
        sx = s(tp.x0)
        μx += sx.EI
        ∇μx += sx.∇EI
    end

    return μx, ∇μx, stderr_μx, stderr_∇μx
end

function simulate_trajectory_deterministic(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    testfn::TestFunction,
    )
    deepcopy_s = Base.deepcopy(s)

    # Rollout trajectory
    T = Trajectory(deepcopy_s, tp.x0, tp.h)
    rollout_deterministic!(T, tp.lbs, tp.ubs;
        xstarts=xstarts, testfn=testfn
    )

    return α(T), ∇αd(T)
end