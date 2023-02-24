# using Distributions
# using LinearAlgebra

include("lazy_struct.jl")
include("radial_basis_functions.jl")

# ------------------------------------------------------------------
# Operations on GP/RBF surrogates
# ------------------------------------------------------------------
struct RBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    fK::Cholesky
    y::Vector{Float64}
    c::Vector{Float64}
    σn2::Float64
    ymean::Float64
end

@everywhere struct RBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    fK::Cholesky
    y::Vector{Float64}
    c::Vector{Float64}
    σn2::Float64
    ymean::Float64
end

function plot1D(s::RBFsurrogate; domain)
    p = scatter(s.X', s.y .+ s.ymean, label="Observations")
    plot!(
        p,
        domain,
        [s([x]).μ + s.ymean for x in domain],
        ribbons=2*[s([x]).σ for x in domain],
        label="μ±2σ"
    )
    return p
end

function plot1DEI(s::RBFsurrogate; domain)
    p = plot(
        domain,
        [s([x]).EI for x in domain],
        label="EI"
    )
    return p
end

function fit_surrogate(ψ::RBFfun, X::Matrix{Float64}, f::Function; σn2=1e-6)
    d, N = size(X)
    K = eval_KXX(ψ, X; σn2=σn2)
    fK = cholesky(Hermitian(K))
    y = [f(X[:,j]) for j = 1:N]
    ymean = mean(y)
    y = y .- ymean
    c = fK\y
    return RBFsurrogate(ψ, X, K, fK, y, c, σn2, ymean)
end

function fit_surrogate(ψ::RBFfun, X::Matrix{Float64}, y::Vector{Float64}; σn2=1e-6)
    d, N = size(X)
    K = eval_KXX(ψ, X, σn2=σn2)
    fK = cholesky(Hermitian(K))
    ymean = mean(y)
    y = y .- ymean
    c = fK\y
    return RBFsurrogate(ψ, X, K, fK, y, c, σn2, ymean)
end

@everywhere function update_surrogate(s::RBFsurrogate, x::Vector{Float64}, f::Function)
    X = hcat(s.X, x)
    recover_y = s.y .+ s.ymean
    y = vcat(recover_y, f(x))
    ymean = mean(y)
    y = y .- ymean
    KxX = eval_KxX(s.ψ, x, s.X)
    K = [s.K  KxX
         KxX' eval_KXX(s.ψ, reshape(x, length(x), 1), σn2=s.σn2)]
    fK = cholesky(Hermitian(K))
    c = fK\y
    return RBFsurrogate(s.ψ, X, K, fK, y, c, s.σn2, ymean)
end

# function update_surrogate(s::RBFsurrogate, x::Vector{Float64}, ys::Vector{Float64})
#     X = hcat(s.X, x)
#     y = ys
#     KxX = eval_KxX(s.ψ, x, s.X)
#     K = [s.K  KxX
#          KxX' eval_KXX(s.ψ, reshape(x, length(x), 1), σn2=s.σn2)]
#     fK = cholesky(Hermitian(K))
#     c = fK\y
#     return RBFsurrogate(s.ψ, X, K, fK, y, c, s.σn2)
# end

@everywhere function update_surrogate(s::RBFsurrogate, x::Vector{Float64}, ys::Float64)
    X = hcat(s.X, x)
    recover_y = s.y .+ s.ymean
    y = vcat(recover_y, ys)
    ymean = mean(y)
    y = y .- ymean
    KxX = eval_KxX(s.ψ, x, s.X)
    K = [s.K  KxX
         KxX' eval_KXX(s.ψ, reshape(x, length(x), 1), σn2=s.σn2)]
    fK = cholesky(Hermitian(K))
    c = fK\y
    return RBFsurrogate(s.ψ, X, K, fK, y, c, s.σn2, ymean)
end

"""
@TODO: Investigate using automatic differentiation to compute the gradient of the
analytic terms.
"""

@everywhere function eval(s::RBFsurrogate, x::Vector{Float64}, ymin::Real)
    sx = LazyStruct()
    set(sx, :s, s)
    set(sx, :x, x)
    set(sx, :ymin, ymin)

    d, N = size(s.X)

    sx.kx = () -> eval_KxX(s.ψ, x, s.X)
    sx.∇kx = () -> eval_∇KxX(s.ψ, x, s.X)

    sx.μ = () -> dot(sx.kx, s.c)
    sx.∇μ = () -> sx.∇kx * s.c
    sx.Hμ = function()
        H = zeros(d, d)
        for j = 1:N
            H += s.c[j] * eval_Hk(s.ψ, x-s.X[:,j])
        end
        return H
    end

    sx.w = () -> s.fK\sx.kx
    sx.Dw = () -> s.fK\(sx.∇kx')
    sx.∇w = () -> sx.Dw'
    sx.σ = () -> sqrt(s.ψ(0) - dot(sx.kx', sx.w))
    sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
    sx.Hσ = function()
        H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
        w = sx.w
        for j = 1:N
            H -= w[j] * eval_Hk(s.ψ, x-s.X[:,j])
        end
        H /= sx.σ
        return H
    end

    sx.z = () -> (ymin - sx.μ) / sx.σ
    sx.∇z = () -> (-sx.∇μ - sx.z*sx.∇σ) / sx.σ
    sx.Hz = () -> Hermitian((-sx.Hμ + (sx.∇μ*sx.∇σ' + sx.∇σ*sx.∇μ')/sx.σ -
        sx.z*(sx.Hσ - 2/sx.σ*sx.∇σ*sx.∇σ')) / sx.σ)

    sx.Φz = () -> Distributions.normcdf(sx.z)
    sx.ϕz = () -> Distributions.normpdf(sx.z)
    sx.g = () -> sx.z * sx.Φz + sx.ϕz

    sx.EI = () -> sx.σ*sx.g
    sx.∇EI = () -> sx.g*sx.∇σ + sx.σ*sx.Φz*sx.∇z
    sx.HEI = () -> Hermitian(sx.Hσ*sx.g +
        sx.Φz*(sx.∇σ*sx.∇z' + sx.∇z*sx.∇σ' + sx.σ*sx.Hz) +
        sx.σ*sx.ϕz*sx.∇z*sx.∇z')

    # Optimizing expected improvement is tricky in regions where EI is
    # exponentially small -- we have to have a reasonable starting
    # point to get going.  For negative z values, we rewrite g(z) = G(-z)
    # in terms of the Mills ratio R(z) = Q(z)/ϕ(z) where Q(z) is the
    # complementary CDF.  Then G(z) = H(z) ϕ(z) where H(z) = 1-zR(z).
    # For sufficiently large R, the Mills ratio can be computed by a
    # generalized continued fraction due to Laplace:
    #   R(z) = 1/z+ 1/z+ 2/z+ 3/z+ ...
    # We rewrite this as
    #   R(z) = W(z)/(z W(z)+1) where W(z) = z + 2/z+ 3/z+ ...
    # Using this definition, we have
    #   H(z) = 1/(1+z W(z))
    #   log G(z) = -log(w+zW(z)) + normlogpdf(z)
    #   [log G(z)]' = -Q(z)/G(z) = -W(z)
    #   [log G(z)]'' = 1 + zW(z) - W(z)^2
    # The continued fraction doesn't converge super-fast, but that is
    # almost surely fine for what we're doing here.  If needed, we could
    # do a similar manipulation to get an optimized rational approximation
    # to W from Cody's 1969 rational approximations to erfc.  Or we could
    # use a less accurate approximation -- the point of getting the tails
    # right is really to give us enough inormation to climb out of the flat
    # regions for EI.

    sx.WQint = function()
        z = -sx.z
        u = z
        for k = 500:-1:2
            u = k/(z+u)
        end
        z + u
    end

    sx.logEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            log(sx.σ) - log(1-z*W) + Distributions.normlogpdf(z)
        else
            log(sx.σ) + log(sx.g)
        end
    end

    sx.∇logEI = function()
        z = sx.z
        if z < -1.0
            sx.∇σ/sx.σ + sx.WQint*sx.∇z
        else
            sx.∇σ/sx.σ + sx.Φz/sx.g*sx.∇z
        end
    end

    sx.HlogEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            HlogG = 1.0-(z+W)*W
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        else
            W = sx.Φz/sx.g
            HlogG = (sx.ϕz-sx.Φz*sx.Φz/sx.g)/sx.g
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        end
    end
    
    return sx
end

@everywhere eval(s::RBFsurrogate, x::Vector{Float64}) = eval(s, x, minimum(s.y))
@everywhere (s::RBFsurrogate)(x::Vector{Float64}) = eval(s, x)

@everywhere function gp_draw(s::RBFsurrogate, xloc; stdnormal)
    sx = s(xloc)
    return sx.μ + sx.σ*stdnormal
end

"""
The initial sample from our trajectory needs a gradient sample, so this
should only be used on the initial sample. Subsequent samples should use
the multi-output surrogate gp_draw function.
"""
function dgp_draw(sur::RBFsurrogate, xloc; dim, stdnormal)
    sx = sur(xloc)
    m = [sx.μ, sx.∇μ...]
    Ksx = eval_DKxX(sur.ψ, xloc, sur.X, D=dim)
    Kss = eval_Dk(sur.ψ, 0*xloc; D=dim)
    Kxx = eval_DKXX(sur.ψ, sur.X, D=dim)
    K = Kss - Ksx*(Kxx\Ksx')
    L = cholesky(Matrix(Hermitian(K)), Val(true), check = false).U'
    # L = [sx.σ  sx.∇σ';
    #      sx.∇σ sx.Hσ]
    sample = m + L*stdnormal
    f, ∇f = sample[1], sample[2:end]
    return f, ∇f
end

# ------------------------------------------------------------------
# Operations on GP/RBF surrogate derivatives wrt node positions
# ------------------------------------------------------------------
struct δRBFsurrogate
    s::RBFsurrogate
    X::Matrix{Float64}
    K::Matrix{Float64}
    y::Vector{Float64}
    c::Vector{Float64}
end

@everywhere struct δRBFsurrogate
    s::RBFsurrogate
    X::Matrix{Float64}
    K::Matrix{Float64}
    y::Vector{Float64}
    c::Vector{Float64}
end

function fit_δsurrogate(s::RBFsurrogate, δX::Matrix{Float64}, ∇f::Function)
    d, N = size(s.X)
    δK = eval_δKXX(s.ψ, s.X, δX)
    δy = [dot(∇f(s.X[:,j]), δX[:,j]) for j=1:N]
    δc = s.fK \ (δy - δK*s.c)
    return δRBFsurrogate(s, δX, δK, δy, δc)
end

@everywhere function update_δsurrogate(us::RBFsurrogate, δs::δRBFsurrogate, 
    δx::Vector{Float64}, ∇y::Vector{Float64})
    d, N = size(us.X)
    δX = hcat(δs.X, δx)
    δK = eval_δKXX(us.ψ, us.X, δX)
    δy = vcat(δs.y, dot(∇y, δx))
    δc = us.fK\(δy - δK*us.c)
    return δRBFsurrogate(us, δX, δK, δy, δc)
end

@everywhere function eval(δs :: δRBFsurrogate, sx, δymin)
    δsx = LazyStruct()
    set(δsx, :sx, sx)
    set(δsx, :δymin, δymin)

    s = δs.s
    x = sx.x
    d, N = size(s.X)

    δsx.kx  = () -> eval_δKxX(s.ψ, x, s.X, δs.X)
    δsx.∇kx = () -> eval_δ∇KxX(s.ψ, x, s.X, δs.X)

    δsx.μ  = () -> δsx.kx'*s.c + sx.kx'*δs.c
    δsx.∇μ = () -> δsx.∇kx*s.c + sx.∇kx*δs.c

    δsx.σ  = () -> (-2*δsx.kx'*sx.w + sx.w'*(δs.K*sx.w)) / (2*sx.σ)
    δsx.∇σ = () -> (-δsx.∇kx*sx.w - sx.∇w*δsx.kx + sx.∇w*(δs.K*sx.w)-δsx.σ*sx.∇σ)/sx.σ

    δsx.z  = () -> (δymin-δsx.μ-sx.z*δsx.σ)/sx.σ
    δsx.∇z = () -> (-δsx.∇μ-sx.∇z*δsx.σ-sx.z*δsx.∇σ)/sx.σ - δsx.z/sx.σ*sx.∇σ

    δsx.EI  = () -> sx.g*δsx.σ + sx.σ*sx.Φz*δsx.z
    δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(δsx.z*sx.∇σ + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.ϕz*δsx.z*sx.∇z

    δsx
end


@everywhere function eval(δs :: δRBFsurrogate, sx)
    ymin, j_ymin = findmin(δs.s.y)
    δymin = δs.y[j_ymin]
    eval(δs, sx, δymin)
end


@everywhere (δs :: δRBFsurrogate)(sx) = eval(δs, sx)


@everywhere function evalf(δs :: δRBFsurrogate, sx, δymin, fantasy_ndx)
    # println("Eval with fantasy ndx")
    δsx = LazyStruct()
    set(δsx, :sx, sx)
    set(δsx, :δymin, δymin)

    s = δs.s
    x = sx.x
    d, N = size(s.X)

    δsx.kx  = () -> eval_δKxX(s.ψ, x, s.X, δs.X)
    # δsx.kx = function()
    #     Xknown = s.X[:, 1:fantasy_ndx-1]
    #     Xfantasy = s.X[:, fantasy_ndx:end]
    #     δXfantasy = δs.X[:, fantasy_ndx:end]
    #     known_kx = eval_KxX(s.ψ, x, Xknown)*0
    #     fantasy_kx = eval_δKxX(s.ψ, x, Xfantasy, δXfantasy)
    #     return vcat(known_kx, fantasy_kx)
    # end

    δsx.∇kx = () -> eval_δ∇KxX(s.ψ, x, s.X, δs.X)
    # δsx.∇kx = function()
    #     Xknown = s.X[:, 1:fantasy_ndx-1]
    #     Xfantasy = s.X[:, fantasy_ndx:end]
    #     δXfantasy = δs.X[:, fantasy_ndx:end]
    #     known_∇kx = eval_∇KxX(s.ψ, x, Xknown)
    #     fantasy_∇kx = eval_δ∇KxX(s.ψ, x, Xfantasy, δXfantasy)
    #     # println("Known ∇kx: $(known_∇kx) -- Fantasy ∇kx: $(fantasy_∇kx)")
    #     if fantasy_ndx == size(s.X, 2) + 1
    #         # println("No fantasy yet: $(known_∇kx)")
    #         return known_∇kx
    #     end
    #     # println("Return: $(hcat(known_∇kx, fantasy_∇kx))")
    #     return hcat(known_∇kx, fantasy_∇kx)
    # end

    δsx.μ  = () -> δsx.kx'*s.c + sx.kx'*δs.c
    δsx.∇μ = () -> δsx.∇kx*s.c + sx.∇kx*δs.c

    δsx.σ  = () -> (-2*δsx.kx'*sx.w + sx.w'*(δs.K*sx.w)) / (2*sx.σ)
    δsx.∇σ = () -> (-δsx.∇kx*sx.w - sx.∇w*δsx.kx + sx.∇w*(δs.K*sx.w)-δsx.σ*sx.∇σ)/sx.σ

    δsx.z  = () -> (δymin-δsx.μ-sx.z*δsx.σ)/sx.σ
    δsx.∇z = () -> (-δsx.∇μ-sx.∇z*δsx.σ-sx.z*δsx.∇σ)/sx.σ - δsx.z/sx.σ*sx.∇σ

    δsx.EI  = () -> sx.g*δsx.σ + sx.σ*sx.Φz*δsx.z
    δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(δsx.z*sx.∇σ + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.ϕz*δsx.z*sx.∇z

    δsx
end


@everywhere function evalf(δs :: δRBFsurrogate, sx, fantasy_ndx)
    ymin, j_ymin = findmin(δs.s.y)
    δymin = δs.y[j_ymin]
    evalf(δs, sx, δymin, fantasy_ndx)
end


@everywhere (δs :: δRBFsurrogate)(sx, fantasy_ndx) = evalf(δs, sx, fantasy_ndx)

# ------------------------------------------------------------------
# Operations on multi-output GP/RBF surrogate
# ------------------------------------------------------------------
"""
MultiOutputRBFSurrogate makes the assumption that the vector y contains
function values and derivatives, i.e. y = [f(x1), f(x2), ..., f(xN),
∇f(x1), ∇f(x2), ..., ∇f(xN)]. Therefore, we must specify the start
index of the derivatives in the vector y.
"""
struct MultiOutputRBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    fK::Cholesky
    y::Vector{Float64}
    c::Vector{Float64}
    ∇xndx::Int64
    ∇yndx::Int64
end

@everywhere struct MultiOutputRBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    fK::Cholesky
    y::Vector{Float64}
    c::Vector{Float64}
    ∇xndx::Int64
    ∇yndx::Int64
end

function fit_multioutput_surrogate(ψ::RBFfun, X::Matrix{Float64},
    y::Vector{Float64}; ∇xndx::Int64, ∇yndx::Int64, σn2=1e-6)
    d, N = size(X)
    K = eval_mixed_KXX(ψ, X; j_∇=∇xndx, σn2=σn2)
    fK = cholesky(Hermitian(K))
    c = fK\y
    return MultiOutputRBFsurrogate(ψ, X, K, fK, y, c, ∇xndx, ∇yndx)
end

@everywhere function update_multioutput_surrogate(ms::MultiOutputRBFsurrogate, x::Vector{Float64},
    y::Float64, ∇y::Vector{Float64}, σn2=1e-6)
    d, N = size(ms.X)
    X = hcat(ms.X, x)

    # Ksx = eval_mixed_KxX(ms.ψ, ms.X, x; j_∇=ms.∇xndx)
    # K = [ms.K  Ksx';
    #      Ksx   eval_mixed_Kxx(ψ, x)]
    K = eval_mixed_KXX(ψ, X; j_∇=ms.∇xndx, σn2=σn2)
    fK = cholesky(Hermitian(K))

    yprev, ∇yprev = ms.y[1:ms.∇yndx-1], ms.y[ms.∇yndx:end]
    yprev = vcat(yprev, y)
    ∇yprev = vcat(∇yprev, ∇y)
    y = vcat(yprev, ∇yprev)

    ∇yndx = length(yprev) + 1
    c = fK\y
    return MultiOutputRBFsurrogate(ms.ψ, X, K, fK, y, c, ms.∇xndx, ∇yndx)
end


@everywhere function eval(ms::MultiOutputRBFsurrogate, x::Vector{Float64}, ymin::Real)
    msx = LazyStruct()
    set(msx, :ms, ms)
    set(msx, :x, x)
    set(msx, :ymin, ymin)

    d, N = size(ms.X)

    msx.kx = () -> eval_mixed_KxX(ms.ψ, ms.X, x; j_∇=ms.∇xndx)
    msx.μ = () -> msx.kx * ms.c
    
    return msx
end

@everywhere function eval(s::MultiOutputRBFsurrogate, x::Vector{Float64})
    y = s.y[1:s.∇yndx-1]
    return eval(s, x, minimum(y))
end
@everywhere (s::MultiOutputRBFsurrogate)(x::Vector{Float64}) = eval(s, x)

"""
Given a multi-output GP surrogate and a point x, draw a sample from the
posterior distribution of the function value and its gradient at x.
"""

@everywhere function gp_draw(ms::MultiOutputRBFsurrogate, xloc; stdnormal)
    msx = ms(xloc)
    m = msx.μ
    Ksx = eval_mixed_KxX(ms.ψ, ms.X, xloc; j_∇=ms.∇xndx)
    Kss = eval_mixed_Kxx(ms.ψ, xloc)
    Kxx = eval_mixed_KXX(ms.ψ, ms.X; j_∇=ms.∇xndx)
    K = Kss - Ksx*(Kxx\Ksx')
    L = cholesky(Matrix(Hermitian(K)), Val(true), check = false).U'
    sample = m + L*stdnormal
    f, ∇f = sample[1], sample[2:end]
    return f, ∇f
end

# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function log_likelihood(s :: RBFsurrogate)
    n = size(s.X)[2]
    -s.y'*s.c/2 - sum(log.(diag(s.fK.L))) - n*log(2π)/2
end


function δlog_likelihood(s :: RBFsurrogate, δθ)
    δK = eval_Dθ_KXX(s.ψ, s.X, δθ)
    (s.c'*δK*s.c - tr(s.fK\δK))/2
end


function ∇log_likelihood(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood(s, δθ)
    end
    ∇L
end

function log_likelihood_v(s :: RBFsurrogate)
    n = size(s.X)[2]
    α = s.y'*s.c/n
    -n/2*(1.0 + log(α) + log(2π)) - sum(log.(diag(s.fK.L)))
end


function δlog_likelihood_v(s :: RBFsurrogate, δθ)
    n = size(s.X)[2]
    c = s.c
    y = s.y
    δK = eval_Dθ_KXX(s.ψ, s.X, δθ)
    n/2*(c'*δK*c)/(c'*y) - tr(s.fK\δK)/2
end


function ∇log_likelihood_v(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood_v(s, δθ)
    end
    ∇L
end

function optimize_hypers_optim(s::RBFsurrogate, ψconstructor)    
    # θ contains kernel variance and lengthscale parameters [σf, l]
    function f(θ)
        scaled_ψ = kernel_scale(ψconstructor, θ)
        ψref = kernel_matern52([1.])
        lsur = fit_surrogate(scaled_ψ, s.X, s.y)
        Lref = log_likelihood(fit_surrogate(ψref, s.X, s.y))
        return log_likelihood(lsur)/Lref
    end

    # res = optimize(θ -> log_likelihood(fit_surrogate(θ, s.X, s.y))/Lref,
    #                s.ψ.θ, LBFGS(), Optim.Options(show_trace=true))
    θinit = [1., s.ψ.θ[1]]
    lowerbounds = [1e-3, 1e-3]
    upperbounds = [10, 10]
    res = optimize(f, lowerbounds, upperbounds, θinit)

    return res
end

function optimize_hypers(θ, kernel_constructor, X, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end

function optimize_hypers(θ, kernel_constructor, X, sur::RBFsurrogate, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, sur, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, sur, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end


function optimize_hypers_v(θ, kernel_constructor, X, f;
                           Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                           monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)

    Lref = log_likelihood_v(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood_v(s)/Lref
    ∇g(s) = ∇log_likelihood_v(s)/Lref

    θ0, s = tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                   Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                   monitor=monitor)
    α = s.c'*s.y/(size(s.X)[2])
    θ = vcat([α], θ0)
    rbf = kernel_scale(kernel_constructor, θ)
    θ, fit_surrogate(rbf, X, f)
end