using Optim
using Sobol

include("radial_basis_surrogates.jl")


function log_ei_solve(s::RBFsurrogate, lbs::Vector{Float64}, ubs::Vector{Float64}
    , xstart::Vector{Float64})
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

function multistart_ei_solve(s::RBFsurrogate, lbs::Vector{Float64},
    ubs::Vector{Float64}, xstarts::Matrix{Float64}; iters::Int = 100)
    xnext, trash = log_ei_solve(s, lbs, ubs, xstarts[:, 1])
    count = 1
    
    # println("sxnext.EI: $(s(xnext).EI)")
    while (s(xnext).EI ≈ 0 || isnan(s(xnext).EI)) && count < iters
        try
            xnext, trash = log_ei_solve(s, lbs, ubs, xstarts[:, count])
            count += 1
        catch e
            continue
        end
    end

    return xnext
end