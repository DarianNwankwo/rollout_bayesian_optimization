include("covariance_matrix.jl")

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

"""
Generate a batch of N points inbounds relative to the lowerbounds and
upperbounds
"""
function generate_batch(N; lbs, ubs, ϵinterior=1e-2)
    s = SobolSeq(lbs, ubs)
    B = reduce(hcat, next!(s) for i = 1:N)
    # Concatenate a few points to B near the bounds in each respective dimension.
    # lbs_interior = lbs .+ ϵinterior
    # ubs_interior = ubs .- ϵinterior
    # B = hcat(B, lbs_interior)
    # B = hcat(B, ubs_interior)
    return B
end

function filter_batch(X, x; tol=1e-2)
    to_keep = []
    for j in 1:size(X, 2)
        if norm(X[:,j] - x) > tol
            push!(to_keep, j)
        end
    end
    return X[:, to_keep]
end

function generate_batch(N, X; lbs, ubs)
    s = SobolSeq(lbs, ubs)
    B = reduce(hcat, next!(s) for i = 1:N*2)
    B = convert(Matrix{Float64}, filter(x -> !(x in X), B)')
    return B[:, 1:N]
end

function centered_fd(f, u, du, h)
    (f(u+h*du)-f(u-h*du))/(2h)
end

function update_λ(λ, ∇g)
    k = ceil(log10(norm(∇g)))
    return λ * 10. ^ (-k)
end

function update_x(x; λ, ∇g, lbs, ubs)
    # λ = update_λ(λ, ∇g)
    x = x .+ λ*∇g
    x = max.(x, lbs)
    x = min.(x, ubs)
    return x
end

"""
This assumes stochastic gradient ascent

x0: input to function g
∇g: gradient of function g
λ: learning rate
β1, β2, ϵ
m: first moment estimate
v: second moment estimate
"""
function update_x_adam(x0; ∇g,  λ, β1, β2, ϵ, m, v, lbs, ubs)
    # Print the dimension and name of each parameter in this function
    v = β2 * v + (1 - β2) * ∇g.^2  # Update second moment estimate
    m_hat = m / (1 - β1)  # Correct for bias in first moment estimate
    v_hat = v / (1 - β2)  # Correct for bias in second moment estimate
    x = x0 + λ * m_hat ./ (sqrt.(v_hat) .+ ϵ)  # Compute updated position
    x = max.(x, lbs)
    x = min.(x, ubs)
    return x, m, v  # Return updated position and updated moment estimates
end

function stochastic_gradient_ascent_adam(x0;
    λ=0.01, β1=0.9, β2=0.999, ϵ=1e-8, ftol=1e-6, gtol=1e-6, objective=:EI, varred=true,
    sur, max_sgd_iters, lbs, ubs, mc_iters, lds_rns, horizon, max_counter)
    m = zeros(size(x0))
    v = zeros(size(x0))
    xstart = x0

    objs, grads, obj_old = [-Inf], [[-Inf]], -Inf
    iters = 1
    counter = 0
    μx, ∇μx = 0., zeros(size(x0))

    for epoch in 1:max_sgd_iters
        iters = epoch

        # Compute stochastic estimates of function and gradient
        try
            μx, ∇μx = simulate_trajectory(sur;
                mc_iters=mc_iters, rnstream=lds_rns, lbs=lbs, ubs=ubs, x0=x0, h=horizon, objective=objective
            )
        catch e
            return (start=xstart, finish=xstart, final_obj=-Inf, final_grad=[-Inf], iters=iters, success=false)
        end

        # Update position and moment estimates
        x0, m, v = update_x_adam(x0; ∇g=∇μx, λ=λ, β1=β1, β2=β2, ϵ=ϵ, m=m, v=v, lbs=lbs, ubs=ubs)

        # Store computed objective
        if varred
            μx += sur(x0).EI
        end
        push!(objs, μx)
        push!(grads, ∇μx)

        # Check for convergence: gradient is approx 0,
        if abs(objs[end] - obj_old) < ftol
            # println("Objective is small")
            break
        end

        if norm(grads[end]) < gtol
            # println("Gradient is small")
            break
        end
        
        counter += objs[end] - obj_old < 0
        if counter >= max_counter
            # println("Counter is large")
            break
        end 

        obj_old = objs[end]
    end

    return (start=xstart, finish=x0, final_obj=objs[end], final_grad=grads[end], iters=iters, success=true)
end

function stochastic_gradient_ascent_adam_efficient(x0;
    λ=0.01, β1=0.9, β2=0.999, ϵ=1e-8, ftol=1e-6, gtol=1e-6, objective=:EI, varred=true,
    sur, max_sgd_iters, lbs, ubs, mc_iters, lds_rns, horizon, max_counter)
    m = zeros(size(x0))
    v = zeros(size(x0))
    xstart = x0

    objs, grads, obj_old = [-Inf], [[-Inf]], -Inf
    iters = 1
    counter = 0
    μx, ∇μx = 0., zeros(size(x0))

    mhistory, d = size(sur.X)
    psm = PreallocatedSymmetricMatrix{Float64}(mhistory, horizon, d)

    for epoch in 1:max_sgd_iters
        iters = epoch

        # Compute stochastic estimates of function and gradient
        try
            μx, ∇μx = simulate_trajectory(sur;
                mc_iters=mc_iters, rnstream=lds_rns, lbs=lbs, ubs=ubs, x0=x0, h=horizon, objective=objective
            )
        catch e
            return (start=xstart, finish=xstart, final_obj=-Inf, final_grad=[-Inf], iters=iters, success=false)
        end

        # Update position and moment estimates
        x0, m, v = update_x_adam(x0; ∇g=∇μx, λ=λ, β1=β1, β2=β2, ϵ=ϵ, m=m, v=v, lbs=lbs, ubs=ubs)

        # Store computed objective
        if varred
            μx += sur(x0).EI
        end
        push!(objs, μx)
        push!(grads, ∇μx)

        # Check for convergence: gradient is approx 0,
        if abs(objs[end] - obj_old) < ftol
            # println("Objective is small")
            break
        end

        if norm(grads[end]) < gtol
            # println("Gradient is small")
            break
        end
        
        counter += objs[end] - obj_old < 0
        if counter >= max_counter
            # println("Counter is large")
            break
        end 

        obj_old = objs[end]
    end

    return (start=xstart, finish=x0, final_obj=objs[end], final_grad=grads[end], iters=iters, success=true)
end

function stochastic_gradient_ascent_vanilla(x0;
    λ=0.01, ftol=1e-6, gtol=1e-6, objective=:EI, sur, max_sgd_iters, lbs, ubs, mc_iters, lds_rns, horizon)
    xstart = x0

    objs, grads, obj_old = [], [], -Inf
    iters = 1
    μx, ∇μx = 0., zeros(size(x0))

    for epoch in 1:max_sgd_iters
        iters = epoch

        # Compute stochastic estimates of function and gradient
        try
            μx, ∇μx = simulate_trajectory(sur;
                mc_iters=mc_iters, rnstream=lds_rns, lbs=lbs, ubs=ubs, x0=x0, h=horizon, objective=objective
            )
        catch e
            return (start=xstart, finish=xstart, final_obj=-Inf, final_grad=[-Inf], iters=iters, success=false)
        end

        # Update position and moment estimates
        x0 = update_x(x0; λ=λ, ∇g=∇μx, lbs=lbs, ubs=ubs)
        # Store computed objective
        push!(objs, μx)
        push!(grads, ∇μx)

        # Check for convergence
        if abs(objs[end] - obj_old) < ftol || norm(grads[end]) < gtol
            break
        end

        obj_old = objs[end]
    end

    return (start=xstart, finish=x0, final_obj=objs[end], final_grad=grads[end], iters=iters, success=true)
end


function ei(μ, σ, fbest)
    z = (fbest - μ) / σ
    Φz = Distributions.normcdf(z)
    ϕz = Distributions.normpdf(z)
    return σ*(z*Φz + ϕz)
end

function poi(μ, σ, fbest)
    z = (fbest - μ) / σ
    Φz = Distributions.normcdf(z)
    return Φz
end

function measure_gap(sur::RBFsurrogate, fbest)
    gaps = []
    init_mini = sur.y[1] .+ sur.ymean
    maximum_possible_reduction = init_mini - fbest
    
    for i in 1:length(sur.y)
        cur_mini = minimum(sur.y[1:i]) .+ sur.ymean
        gap = (init_mini - cur_mini) / maximum_possible_reduction
        push!(gaps, gap)
    end
    
    return gaps
end

function create_filename(dict::Dict{String, Any})
    function_name = dict["function-name"]
    function_args = dict["function-args"]
    h = dict["horizon"]
    mc_iters = dict["mc-samples"]
    sgd_iters = dict["sgd-iters"]
    b = dict["batch-size"]
    filename = string(function_name, "_", function_args, "_bayesopt_horizon", h, "_mc", mc_iters, "_sgd", sgd_iters, "_batchsize", b, ".csv")
    return filename
end
