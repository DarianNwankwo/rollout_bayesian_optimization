using Distributions
using Sobol

"""
Generate low discrepancy Sobol sequence of uniform random variables
"""
function gen_uniform(samples; dim=1)
    sobol = SobolSeq(zeros(dim), ones(dim))
    S = zeros(dim, samples)
    
    for j in 1:samples
        S[:,j] = next!(sobol)
    end
    
    S
end

function is_odd(num)
    return mod(num,2) == 1
end

"""
Transforms an even-sized multivariate uniform distribution to be normally
distributed with mean 0 and variance 1.
"""
function box_muller_transform(S)
    dim, samples = size(S)
    N = zeros(dim, samples)
    
    for j in 1:samples
        y = zeros(dim)
        x = S[:,j]
        
        for i in 1:dim
            if is_odd(i)
                y[i] = sqrt(-2log10(x[i]))*cos(2π*x[i+1])
            else
                y[i] = sqrt(-2log10(x[i-1]))*sin(2π*x[i])
            end
        end
        
        N[:,j] = y
    end
    
    N
end

"""
Produces a sequence of standard normally distributed values in 1D
"""
function uniform1d_to_normal(samples)
    uniform2d = gen_uniform(samples, dim=2)
    normal2d = box_muller_transform(uniform2d)
    marginal_normal = normal2d[1,:]
    
    marginal_normal
end

"""
Generate a low discrepancy multivariate normally distributed sequence
for monte carlo simulation of rollout acquisition functions with a max
horizon of h. The resulting tensor will be of size Mx(D+1)xH, where M
is the number of Monte Carlo iterations, D is the dimension of the
input, and H is our max horizon.
"""
function gen_low_discrepancy_sequence(samples, dim, horizon)
    offset = isodd(dim+1) ? 1 : 0
    S = gen_uniform(samples*horizon, dim=dim+1+offset)
    N = box_muller_transform(S)
    N = reshape(N, samples, dim+1+offset, horizon)
    
    N[:,1:end-offset,:]
end

function randsample(N, d, lbs, ubs)
    X = zeros(d, N)
    for j = 1:N
        for i = 1:d
            X[i,j] = rand(Uniform(lbs[i], ubs[i]))
        end
    end
    X
end

function stdize(series)
    smax, smin = maximum(series), minimum(series)
    return [(s-smin)/(smax-smin) for s in series]
end

function pairwise_diff_issmall(prev, next; tol=1e-6)
    # If minimizing, prev-next. If maximizing, next-prev
    return abs(prev-next) < tol
end

function pairwise_forward_diff(series)
    N = length(series)
    result = []
    
    for i in 1:N-1
        diff = series[i] - series[i+1]
        push!(result, diff)
    end
    
    result
end

function is_still_increasing(series)
    a, b, c = series[end-2:end]
    return a < b < c
end

function print_divider(char="-"; len=80)
    for _ in 1:len print(char) end
    println()
end

"""
Returns true as long as we need to keep performing steps of our optimization
iteration. As soon as we reach our maximum iteration count we terminate, but
if the pairwise difference is sufficiently large (> tol), we believe we can
keep improving and continue our iteration. If the pairwise difference is not
large, there is a chance we can improve, so we continue. The next iteration
then checks if our rate of change has gone from increasing to decreasing,
w.l.o.g.
"""
function sgd_hasnot_converged(iter, grads, max_iters; tol=1e-4)
    # bpdiff captures the relative improvement 
    return ((!(iter > 3 && pairwise_diff_issmall(grads[end-1:end]..., tol=tol))
           || (iter > 3 && is_still_increasing(grads)))
           && iter < max_iters)
end


function get_maximum_key(datadict)
    key, value = 0., 0.
    
    for pairs in datadict
        if pairs.second[1] > value
            key = pairs.first
            value = pairs.second[1]
        end
    end
    
    return key
end


function inbounds(x0, lbs, ubs)
    return all(lbs .< x0 .< ubs)
end