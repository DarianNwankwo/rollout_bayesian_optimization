include("radial_basis_surrogates.jl")


mutable struct Trajectory
    s::RBFsurrogate
    fs::Union{FantasyRBFsurrogate, Nothing}
    δfs::Union{δRBFsurrogate, Nothing}
    mfs::Union{MultiOutputFantasyRBFsurrogate, Nothing}
    opt_HEI::Union{Matrix{Float64}, Nothing}
    fmin::Union{Float64, Nothing}
    x0::Vector{Float64}
    h::Int
end

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.
"""
function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
    # The ground truth surrogate is zero mean, so when we sample from our GP, we
    # need to add the mean back in when we are updating the surrogate.
    fmin = minimum(s.y) + s.ymean
    d, N = size(s.X)

    ∇ys = [zeros(d) for i in 1:N]
    δX = zeros(d, N)

    fsur = fit_fsurrogate(s, h)
    δsur = fit_δsurrogate(fsur, δX, ∇ys)
    mfsur = fit_multioutput_fsurrogate(sur, h)

    opt_HEI = zeros(d, d)

    return Trajectory(s, fsur, δsur, mfsur, opt_HEI, fmin, x0, h)
end

function reset!(T::Trajectory)
    fmin = minimum(T.s.y) + T.s.ymean
    d, N = size(T.s.X)

    reset_fsurrogate!(T.fs, T.s)
    reset_δsurrogate!(T.δfs, T.fs)
    reset_mfsurrogate!(T.mfs, T.s)

    T.opt_HEI = zeros(d, d)
end