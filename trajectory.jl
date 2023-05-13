include("radial_basis_surrogates_update.jl")


mutable struct Trajectory
    s::RBFsurrogate
    δs::δRBFsurrogate
    ms::MultiOutputRBFsurrogate
    xfs::Matrix{Float64}
    ys::Vector{Float64}
    ∇ys::Matrix{Float64}
    h::Int
    fantasy_ndx::Int
    opt_HEI::Matrix{Float64}
    fopt::Float64
end

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.
"""
function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, fndx::Int; h::Int, fopt::Float64)
    d, N = size(s.X)
    # Initialize base surrogates as placeholders
    δs = δRBFsurrogate(s, zeros(d, N), s.K, s.y, s.c, 0.)
    # δs = δRBFsurrogate(s, s.X, s.K, s.y, s.c)
    ms = MultiOutputRBFsurrogate(
        s.ψ, s.X, s.K, s.fK, s.y, s.c, N+1, length(s.y) + 1, 0., zeros(d)    
    )
    
    # Preallocate memory for trajectory path and samples
    xfs, ys, ∇ys = zeros(d, h+1), zeros(h+1), zeros(d, h+1)
    xfs[:, 1] = x0

    opt_HEI = zeros(d, d)

    return Trajectory(s, δs, ms, xfs, ys, ∇ys, h, fndx, opt_HEI, fopt)
end

mutable struct SmartTrajectory
    s::RBFsurrogate
    δs::δRBFsurrogate
    ms::MultiOutputRBFsurrogate
    xfs::Matrix{Float64}
    ys::Vector{Float64}
    ∇ys::Matrix{Float64}
    h::Int
    fantasy_ndx::Int
    opt_HEI::Matrix{Float64}
    fopt::Float64
end

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.
"""
function SmartTrajectory(s::RBFsurrogate, x0::Vector{Float64}, fndx::Int; h::Int, fopt::Float64)
    d, N = size(s.X)
    # Initialize base surrogates as placeholders
    δs = δRBFsurrogate(s, zeros(d, N), s.K, s.y, s.c, 0.)
    # δs = δRBFsurrogate(s, s.X, s.K, s.y, s.c)
    ms = MultiOutputRBFsurrogate(
        s.ψ, s.X, s.K, s.fK, s.y, s.c, N+1, length(s.y) + 1, 0., zeros(d)    
    )
    
    # Preallocate memory for trajectory path and samples
    xfs, ys, ∇ys = zeros(d, h+1), zeros(h+1), zeros(d, h+1)
    xfs[:, 1] = x0

    opt_HEI = zeros(d, d)

    return Trajectory(s, δs, ms, xfs, ys, ∇ys, h, fndx, opt_HEI, fopt)
end