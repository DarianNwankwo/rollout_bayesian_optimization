using SharedArrays
using Distributed

include("radial_basis_surrogates.jl")


function ei_solve(s::FantasyRBFsurrogate, lbs::Vector{Float64}, ubs::Vector{Float64}
    , xstart::Vector{Float64})
    fun(x) = -s(x).EI
    function fun_grad!(g, x)
        EIx = -s(x).∇EI
        for i in eachindex(EIx)
            g[i] = EIx[i]
        end
    end
    function fun_hess!(h, x)
        HEIx = -s(x).HEI
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

"""
It would be better to do this in parallel.
"""

function multistart_ei_solve(s::FantasyRBFsurrogate, lbs::Vector{Float64},
    ubs::Vector{Float64}, xstarts::Matrix{Float64})::Vector{Float64}
    candidates = []
    
    for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        try
            minimizer, res = ei_solve(s, lbs, ubs, xi)
            push!(candidates, (minimizer, minimum(res)))
        catch e
            println(e)
        end
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    minimizer = candidates[j_mini][1]

    return minimizer
end

function distributed_multistart_ei_solve(s::FantasyRBFsurrogate, lbs::Vector{Float64},
    ubs::Vector{Float64}, xstarts::Matrix{Float64})::Vector{Float64}
    candidate_locations = SharedMatrix{Float64}(size(xstarts, 1), size(xstarts, 2))
    candidate_values = SharedArray{Float64}(size(xstarts, 2))
    
    @sync @distributed for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        try
            minimizer, res = ei_solve(s, lbs, ubs, xi)
            candidate_locations[:, i] = minimizer
            candidate_values[i] = minimum(res)
        catch e
        end
    end
    
    replace!(candidate_values, NaN => Inf)
    mini, j_mini = findmin(candidate_values)
    minimizer = candidate_locations[:, j_mini]

    return minimizer
end