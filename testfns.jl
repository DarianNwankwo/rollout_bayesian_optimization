# https://www.sfu.ca/~ssurjano/optimization.html
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

struct TestFunction
    dim
    bounds
    xopt
    f
    ∇f
end


(f :: TestFunction)(x) = f.f(x)


function tplot(f :: TestFunction)
    if f.dim == 1
        xx = range(f.bounds[1,1], f.bounds[1,2], length=250)
        plot(xx, (x) -> f([x]), label="f(x)")
        scatter!([xy[1] for xy in f.xopt], [f(xy) for xy in f.xopt], label="xopt")
    elseif f.dim == 2
        xx = range(f.bounds[1,1], f.bounds[1,2], length=100)
        yy = range(f.bounds[2,1], f.bounds[2,2], length=100)
        plot(xx, yy, (x,y) -> f([x,y]), st=:contour)
        scatter!([xy[1] for xy in f.xopt], [xy[2] for xy in f.xopt], label="xopt")
        # scatter!([f.xopt[1]], [f.xopt[2]], label="xopt")
    else
        error("Can only plot 1- or 2-dimensional TestFunctions")
    end
end


function TestBraninHoo(; a=1, b=5.1/(4π^2), c=5/π, r=6, s=10, t=1/(8π))
    function f(xy)
        x = xy[1]
        y = xy[2]
        a*(y-b*x^2+c*x-r)^2 + s*(1-t)*cos(x) + s
    end
    function ∇f(xy)
        x = xy[1]
        y = xy[2]
        dx = 2*a*(y-b*x^2+c*x-r)*(-b*2*x+c) - s*(1-t)*sin(x)
        dy = 2*a*(y-b*x^2+c*x-r)
        [dx, dy]
    end
    bounds = [-5.0 10.0 ; 0.0 15.0]
    xopt = ([-π, 12.275], [π, 2.275], [9.42478, 2.475])
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestRosenbrock()
    f(xy) = (1-xy[1])^2 + 100*(xy[2]-xy[1]^2)^2
    ∇f(xy) = [-2*(1-xy[1]) - 400*xy[1]*(xy[2]-xy[1]^2), 200*(xy[2]-xy[1]^2)]
    return TestFunction(2, [-2.0 2.0 ; -1.0 3.0 ], (ones(2),), f, ∇f)
end


function TestRastrigin(n)
    f(x) = 10*n + sum(x.^2 - 10*cos.(2π*x))
    ∇f(x) = 2*x + 20π*sin.(2π*x)
    bounds = zeros(n, 2)
    bounds[:,1] .= -5.12
    bounds[:,2] .=  5.12
    xopt = (zeros(n),)
    return TestFunction(n, bounds, xopt, f, ∇f)
end


function TestAckley(d; a=20.0, b=0.2, c=2π)
    
    function f(x)
        nx = norm(x)
        cx = sum(cos.(c*x))
        -a*exp(-b/sqrt(d)*nx) - exp(cx/d) + a + exp(1)
    end
    
    function ∇f(x)
        nx = norm(x)
        if nx == 0.0
            return zeros(d)
        else
            cx = sum(cos.(c*x))
            dnx = x/nx
            dcx = -c*sin.(c*x)
            (a*b)/sqrt(d)*exp(-b/sqrt(d)*norm(x))*dnx - exp(cx/d)/d*dcx
        end
    end

    bounds = zeros(d,2)
    bounds[:,1] .= -32.768
    bounds[:,2] .=  32.768
    xopt = (zeros(d),)

    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestSixHump()

    function f(xy)
        x = xy[1]
        y = xy[2]
        xterm = (4.0-2.1*x^2+x^4/3)*x^2
        yterm = (-4.0+4.0*y^2)*y^2
        xterm + x*y + yterm
    end

    function ∇f(xy)
        x = xy[1]
        y = xy[2]
        dxterm = (-4.2*x+4.0*x^3/3)*x^2 + (4.0-2.1*x^2+x^4/3)*2.0*x
        dyterm = (8.0*y)*y^2 + (-4.0+4.0*y^2)*2.0*y
        [dxterm + y, dyterm + x]
    end

    # There's a symmetric optimum
    xopt = ([0.089842, -0.712656], [-0.089842, 0.712656])

    return TestFunction(2, [-3.0 3.0 ; -2.0 2.0], xopt, f, ∇f)
end


function TestGramacyLee()
    f(x) = sin(10π*x[1])/(2*x[1]) + (x[1]-1.0)^4
    ∇f(x) = [5π*cos(10π*x[1])/x[1] - sin(10π*x[1])/(2*x[1]^2) + 4*(x[1]-1.0)^3]
    bounds = zeros(1, 2)
    bounds[1,1] = 0.5
    bounds[1,2] = 2.5
    xopt=([0.548563],)
    return TestFunction(1, bounds, xopt, f, ∇f)
end

function TestEasom()
    f(x) = -cos(x[1])*cos(x[2])*exp(-((x[1]-π)^2 + (x[2]-π)^2))

    function ∇f(x)
        function common_subexpression(x)
            c = cos(x[1]) * cos(x[2])
            e = exp(-((x[1] - π)^2 + (x[2] - π)^2))
            term = 2 * (x[1] - π) * cos(x[2]) + 2 * (x[2] - π) * cos(x[1])
            return c, e, term
        end

        c, e, term = common_subexpression(x)
        
        df1 = c * e * term - sin(x[1]) * cos(x[2]) * e
        df2 = c * e * term - sin(x[2]) * cos(x[1]) * e
        
        return [df1, df2]
    end

    bounds = zeros(2, 2)
    bounds[:, 1] = -100.0
    bounds[:, 2] = 100.0
    
    xopt=([π, π],)

    return TestFunction(2, bounds, xopt, f, ∇f)
end


# function TestMichalewicz(d; m=10.0)
# end


function ConstantTestFunction(n=0; lbs::Vector{<:T}, ubs::Vector{<:T}) where T <: Real
    f(x) = n
    ∇f(x) = zeros(length(x))
    xopt = (zeros(length(lbs)),)
    bounds = hcat(lbs, ubs)
    return TestFunction(length(lbs), bounds, xopt, f, ∇f)
end