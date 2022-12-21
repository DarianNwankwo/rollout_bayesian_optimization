using Optim
using Sobol
import Base.~

include("utils.jl")

"""
s: Radial basis function surrogate
δs: Perturbed surrogate
x0: Deterministic start location
xfs: Vector of fantasized vector locations (trajectory path)
ys: Vector of fantasized samples drawn at each step in the trajectory
∇ys: Vector of fantasized gradient samples drawn at each step in the trajectory
h: Horizon of the trajectory
fantasy_ndx: The index associated with the fantasized values
"""
mutable struct Trajectory
    s
    δs
    x0
    xfs
    ys
    ∇ys
    h
    fantasy_ndx
end

mutable struct Trajectory2
    s
    ms # multi-output surrogate
    δs
    x0
    xfs
    ys
    ∇ys
    h
    fantasy_ndx
end

function Trajectory(s::RBFsurrogate, x0::V, h::Int64, fantasy_ndx::Int64) where V<:Vector{Float64}
    # Taking in a copy of the surrogate, but it needs to sample a fantasized value and gradient at
    # x0 and update s and δs
    xfs = Vector{V}()
    ys = V()
    ∇ys = Vector{V}()

    # Compute δs here by passing the identity matrix as our δX
    δs = δRBFsurrogate(s, s.X, s.K, s.y, s.y)

    return Trajectory(s, δs, x0, xfs, ys, ∇ys, h, fantasy_ndx)
end

function Trajectory2(s::RBFsurrogate, x0::V, h::Int64, fantasy_ndx::Int64) where V<:Vector{Float64}
    # Taking in a copy of the surrogate, but it needs to sample a fantasized value and gradient at
    # x0 and update s and δs
    xfs = Vector{V}()
    ys = V()
    ∇ys = Vector{V}()

    # Compute δs here by passing the identity matrix as our δX
    δs = δRBFsurrogate(s, s.X, s.K, s.y, s.y)

    return Trajectory(s, δs, x0, xfs, ys, ∇ys, h, fantasy_ndx)
end

"""
Evaluates the covariance between gradient observations and
all observations. X is expected to contain gradient
observations; whereas Y is expected to contain all
observations.
"""
function covmat_ga(ψ, X, Y)
    m = num_grad_observations = size(X, 2)
    n = num_all_observations = size(Y, 2)
    covmats = []
    
    for i = 1:m
        K = eval_∇k(ψ, X[:, i] - Y[:, 1])
        for j = 2:n
            K = hcat(K, eval_∇k(ψ, X[:, i] - Y[:, j]))
        end
        push!(covmats, K)
    end
    
    
    covmat = covmats[1]
    for i = 2:length(covmats)
        covmat = vcat(covmat, covmats[i])
    end
    return covmat
end

"""
Evaluates the covariance between gradient observations and
all observations. X is expected to contain gradient
observations; whereas Y is expected to contain all
observations. Then transposes it.
"""
function covmat_gat(ψ, X, Y)
    m = num_grad_observations = size(X, 2)
    n = num_all_observations = size(Y, 2)
    covmats = []
    
    for i = 1:m
        K = eval_∇k(ψ, X[:, i] - Y[:, 1])'
        for j = 2:n
            K = vcat(K, eval_∇k(ψ, X[:, i] - Y[:, j])')
        end
        push!(covmats, K)
    end
    
    
    covmat = covmats[1]
    for i = 2:length(covmats)
        covmat = hcat(covmat, covmats[i])
    end
    return covmat
end

"""
Evaluates the covariance between gradient observations and
all observations. X is expected to contain gradient
observations; whereas Y is expected to contain all
observations. Then transposes it.
"""
function covmat_gg(ψ, X)
    m = num_grad_observations = size(X, 2)
    covmats = []
    
    for i = 1:m
        K = eval_Hk(ψ, X[:, i] - Y[:, 1])
        for j = 2:m
            K = hcat(K, eval_Hk(ψ, X[:, i] - Y[:, j]))
        end
        push!(covmats, K)
    end
    
    
    covmat = covmats[1]
    for i = 2:length(covmats)
        covmat = vcat(covmat, covmats[i])
    end
    return covmat
end

"""
Ensure that Y contains the non-gradient observations first
then the gradient observations afterwards.
"""
function covmat_mixed(ψ, X, Y)
    K = [eval_KXX(ψ, Y)     covmat_gat(ψ, X, Y);
         covmat_ga(ψ, X, Y) covmat_gg(ψ, X)]
    return K
end

function ei_solve(s::RBFsurrogate, lbs, ubs, xstart)
    fun(x) = -s(x).logEI
    function fun_grad!(g, x)
        EIx = -s(x).∇logEI
        for i in eachindex(EIx)
            g[i] = EIx[i]
        end
    end
    function fun_hess!(h, x)
        HEIx = -s(x).HlogEI
        for row in 1:size(HEIx, 1)
            for col in 1:size(HEIx, 2)
                h[row, col] = HEIx[row, col]
            end
        end
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(lbs, ubs)
    res = optimize(df, dfc, xstart, IPNewton())

    return Optim.minimizer(res), res
end

function multistart_ei_solve(s::RBFsurrogate, lbs, ubs, xstarts; iters=100)
    # println("s.x size: $(size(s.X))")
    # println("xnext begin: $(xstarts[:, 1])")
    xnext, trash = ei_solve(s, lbs, ubs, xstarts[:, 1])
    count = 1
    
    # println("sxnext.EI: $(s(xnext).EI)")
    while (s(xnext).EI ≈ 0 || isnan(s(xnext).EI)) && count < iters
        # println("Condition for EI Solve: $(s(xnext).EI) -- $(s(xnext).EI ≈ 0)")
        try
            xnext, trash = ei_solve(s, lbs, ubs, xstarts[:, count])
            # println("xnext update: $xnext")
            count += 1
        catch e
            continue
        end
    end

    # println("Condition for EI Solve: $(s(xnext).EI) -- $(s(xnext).EI ≈ 0)")
    # println("Count: $count")
    return xnext
end

function gp_draw(sur::RBFsurrogate, xloc; dim, stdnormal)
    xloc = reshape(xloc, length(xloc), 1)
    sx = sur(xloc)
    m = [sx.μ, sx.∇μ...]
    Ksx = eval_DKxX(sur.ψ, xloc, sur.X, D=dim)
    Kss = eval_DKXX(sur.ψ, xloc, D=dim)
    Kxx = eval_DKXX(sur.ψ, sur.X, D=dim)
    K = Kss - Ksx*(Kxx\Ksx')
    L = cholesky(Matrix(Hermitian(K)), Val(true), check = false).U'
    sample = m + L*stdnormal
    f, ∇f = sample[1], sample[2:end]
    return f, ∇f
end

function rollout(T::Trajectory; RNstream)
    if T.h == 0
        
    end
end

function rollout3!(T::Trajectory, lbs, ubs; RNstream)
    # Initial sample at determinstic start location
    f0, ∇f0 = gp_draw(T.s, T.x0, dim=length(T.x0), stdnormal=RNstream[:,1])
    sx0 = T.s(T.x0);
    HEIbest = sx0.HEI
    δsx0 = -HEIbest \ T.δs(sx0, T.fantasy_ndx).∇EI

    T.s = update_surrogate(T.s, T.x0[:,:], f0)
    T.δs = update_δsurrogate(T.s, T.δs, δsx0, ∇f0)

    push!(T.ys, f0); push!(T.∇ys, ∇f0); push!(T.xfs, T.x0)

    # Perform rollout for the fantasized trajectories
    xstarts = randsample(100, length(∇f0), lbs, ubs)
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.s, lbs, ubs, xstarts)
        
        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.s, xnext, dim=length(xnext), stdnormal=RNstream[:, j])

        # Compute variations in xnext and update surrogate and perturbed surrogate
        sxnext = T.s(xnext)
        δsxnext = -sxnext.HEI \ T.δs(sxnext, T.fantasy_ndx).∇EI
        # println("Fantasized GP Draw: ($(fi), $(∇fi)) -- δsx: $(δsxnext)")

        # Update surrogate and perturbed surrogate
        T.s = update_surrogate(T.s, xnext[:,:], fi)
        T.δs = update_δsurrogate(T.s, T.δs, δsxnext, ∇fi)

        push!(T.ys, fi); push!(T.∇ys, ∇fi); push!(T.xfs, xnext);
    end

    return HEIbest
end

function rollout4!(T::Trajectory, lbs, ubs; RNstream)
    # Initial sample at determinstic start location
    # f0, ∇f0 = gp_draw(T.s, T.x0, dim=length(T.x0), stdnormal=RNstream[:,1])
    f0, ∇f0 = T.s(T.x0).EI, T.s(T.x0).∇EI
    sx0 = T.s(T.x0);
    HEIbest = sx0.HEI
    println(f0, "--", ∇f0, "--", HEIbest)
    δsx0 = -HEIbest \ T.δs(sx0, T.fantasy_ndx).∇EI

    T.s = update_surrogate(T.s, T.x0[:,:], f0)
    T.δs = update_δsurrogate(T.s, T.δs, δsx0, ∇f0)

    push!(T.ys, f0); push!(T.∇ys, ∇f0); push!(T.xfs, T.x0)

    # Perform rollout for the fantasized trajectories
    xstarts = randsample(100, length(∇f0), lbs, ubs)
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext = multistart_ei_solve(T.s, lbs, ubs, xstarts)
        
        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.s, xnext, dim=length(xnext), stdnormal=RNstream[:, j])

        # Compute variations in xnext and update surrogate and perturbed surrogate
        sxnext = T.s(xnext)
        δsxnext = -sxnext.HEI \ T.δs(sxnext, T.fantasy_ndx).∇EI
        # println("Fantasized GP Draw: ($(fi), $(∇fi)) -- δsx: $(δsxnext)")

        # Update surrogate and perturbed surrogate
        T.s = update_surrogate(T.s, xnext[:,:], fi)
        T.δs = update_δsurrogate(T.s, T.δs, δsxnext, ∇fi)

        push!(T.ys, fi); push!(T.∇ys, ∇fi); push!(T.xfs, xnext);
    end

    return HEIbest
end

"""
Computes the sample trajectory using a random number stream

(TODO): We Should be sampling from a multioutput GP and conditioning on those
samples.
"""
function rollout2!(T::Trajectory, testfn::TestFunction; RNstream)
    xnext = T.x0
    fbest = minimum(T.s.y)
    sxnext = T.s(xnext)
    HEIbest = sxnext.HEI
    lbs, ubs = testfn.bounds[:,1], testfn.bounds[:,2]
    seq = SobolSeq(lbs, ubs)
    # Should be T.ms = multi-output surrogate
    fi, ∇fi = gp_draw(T.s, xnext, dim=length(xnext), stdnormal=RNstream[:,1])
    δxnext = -HEIbest \ T.δs(sxnext).∇EI

    T.s = update_surrogate(T.s, xnext[:,:], fi)
    T.δs = update_δsurrogate(T.s, T.δs, δxnext, ∇fi)
    
    push!(T.xfs, xnext)
    push!(T.ys, fi)
    push!(T.∇ys, ∇fi)
    
    for j in 2:T.h
        xnext = multistart_ei_solve(T.s, lbs, ubs, next!(seq))
        fi, ∇fi = gp_draw(T.s, xnext, dim=length(xnext), stdnormal=RNstream[:,j])
        
        # Compute variations in xnext and update surrogate and perturbed surrogate
        sxnext = T.s(xnext)
        δxnext = -sxnext.HEI \ T.δs(sxnext).∇EI
        
        if fi < fbest HEIbest = sxnext.HEI end
        
        T.s = update_surrogate(T.s, xnext[:,:], fi)
        T.δs = update_δsurrogate(T.s, T.δs, δxnext, ∇fi)
        
        push!(T.xfs, xnext)
        push!(T.ys, fi)
        push!(T.∇ys, ∇fi)
    end
    
    return HEIbest
end

function is_rolled_out(T::Trajectory)
    return length(T.xfs) == T.h
end

function sample2(T::Trajectory)
    # Maybe we should raise an error here
    if !is_rolled_out(T) return [] end
    path = [(T.xfs[j], T.ys[j], T.∇ys[j]) for j = 1:T.h]
    return path
end

function sample3(T::Trajectory)
    path = [(T.xfs[j], T.ys[j], T.∇ys[j]) for j in 1:T.h+1]
    return path
end

function best2(T::Trajectory)
    path = sample2(T)
    # Should raise an error here
    if length(path) == 0 return [] end
    # Find the step with the lowest value seen thus far. Index 2
    # corresponds to the function values
    _, minndx = findmin([p[2] for p in path])
    return minndx, path[minndx]
end

function best3(T::Trajectory)
    path = sample3(T)
    _, minndx = findmin([p[2] for p in path])
    return minndx, path[minndx]
end

function α2(T::Trajectory)
    m = T.fantasy_ndx-1 # known sample ending index
    path = sample2(T) # path -> {(x, f(x), ∇f(x)), ...}
    f⁺ = minimum(T.s.y[1:m]) # best value found in history
    b, pathb = best2(T)
    fb = T.ys[b]
    return max(f⁺-fb, 0.)
end

function α3(T::Trajectory)
    m = T.fantasy_ndx-1 # known sample ending index
    path = sample3(T) # path -> {(x, f(x), ∇f(x)), ...}
    f⁺ = minimum(T.s.y[1:m]) # best value found in history
    b, pathb = best3(T)
    fb = T.ys[b]
    return max(f⁺-fb, 0.)
end

function ∇α2(T::Trajectory, HEIbest)
    m = T.fantasy_ndx-1 # known sample ending index
    f⁺ = minimum(T.s.y[1:m]) # best value found in history
    b, pathb = best2(T)
    xb, fb, ∇fb = pathb
    J = HEIbest
    if f⁺ <= fb return zeros(length(T.x0)) end
    return transpose(-∇fb'*J)
end

function ∇α3(T::Trajectory, HEIbest)
    m = T.fantasy_ndx-1 # known sample ending index
    f⁺ = minimum(T.s.y[1:m]) # best value found in history
    b, pathb = best3(T)
    xb, fb, ∇fb = pathb
    J = HEIbest
    if f⁺ <= fb return zeros(length(T.x0)) end
    # return transpose(-∇fb'*J)
    return -∇fb*J
end