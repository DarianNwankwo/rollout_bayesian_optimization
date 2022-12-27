include("radial_basis_surrogates.jl")


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
end

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.
"""
function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, fndx::Int; h::Int)
    d, N = size(s.X)
    # Initialize base surrogates as placeholders
    # δs = δRBFsurrogate(s, zeros(size(s.X)...), s.K, s.y, s.c)
    δs = δRBFsurrogate(s, s.X, s.K, s.y, s.c)
    ms = MultiOutputRBFsurrogate(
        s.ψ, s.X, s.K, s.fK, s.y, s.c, size(s.X, 2) + 1, length(s.y) + 1    
    )
    
    # Preallocate memory for trajectory path and samples
    xfs, ys, ∇ys = zeros(d, h+1), zeros(h+1), zeros(d, h+1)
    xfs[:, 1] = x0

    opt_HEI = zeros(d, d)

    return Trajectory(s, δs, ms, xfs, ys, ∇ys, h, fndx, opt_HEI)
end