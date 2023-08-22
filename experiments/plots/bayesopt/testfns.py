import numpy as np
import matplotlib.pyplot as plt


class TestFunction:
    def __init__(self, dim, bounds, xopt, f, grad_f):
        self.dim = dim
        self.bounds = bounds
        self.xopt = xopt
        self.f = f
        self.grad_f = grad_f

    def __call__(self, x):
        return self.f(x)

    def tplot(self):
        if self.dim == 1:
            xx = np.linspace(self.bounds[0, 0], self.bounds[0, 1], num=250)
            plt.plot(xx, [self([x]) for x in xx], label="f(x)")
            plt.scatter([xy[0] for xy in self.xopt], [self(xy) for xy in self.xopt], label="xopt")
        elif self.dim == 2:
            xx = np.linspace(self.bounds[0, 0], self.bounds[0, 1], num=100)
            yy = np.linspace(self.bounds[1, 0], self.bounds[1, 1], num=100)
            X, Y = np.meshgrid(xx, yy)
            Z = np.array([self([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
            plt.contour(X, Y, Z)
            plt.scatter([xy[0] for xy in self.xopt], [xy[1] for xy in self.xopt], label="xopt")
        else:
            raise ValueError("Can only plot 1- or 2-dimensional TestFunctions")
        plt.legend()
        plt.show()


def test_branin_hoo(a=1, b=5.1 / (4 * np.pi ** 2), c=5 / np.pi, r=6, s=10, t=1 / (8 * np.pi)):
    def f(xy):
        x, y = xy
        return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s

    def grad_f(xy):
        x, y = xy
        dx = 2 * a * (y - b * x ** 2 + c * x - r) * (-b * 2 * x + c) - s * (1 - t) * np.sin(x)
        dy = 2 * a * (y - b * x ** 2 + c * x - r)
        return np.array([dx, dy])

    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    xopt = ([-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475])
    return TestFunction(2, bounds, xopt, f, grad_f)


def test_rosenbrock():
    def f(xy):
        return (1 - xy[0])**2 + 100 * (xy[1] - xy[0]**2)**2
    
    def grad_f(xy):
        return np.array([-2 * (1 - xy[0]) - 400 * xy[0] * (xy[1] - xy[0]**2), 200 * (xy[1] - xy[0]**2)])
    
    bounds = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    xopt = [np.array([1.0, 1.0])]
    
    return TestFunction(2, bounds, xopt, f, grad_f)


def test_rastrigin(n):
    def f(x):
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def grad_f(x):
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    
    bounds = np.zeros((n, 2))
    bounds[:, 0] = -5.12
    bounds[:, 1] = 5.12
    xopt = [np.zeros(n)]
    
    return TestFunction(n, bounds, xopt, f, grad_f)


def test_ackley(d, a=20.0, b=0.2, c=2*np.pi):
    def f(x):
        nx = np.linalg.norm(x)
        cx = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(nx / d)) - np.exp(cx / d) + a + np.exp(1)
    
    def grad_f(x):
        nx = np.linalg.norm(x)
        if nx == 0.0:
            return np.zeros(d)
        else:
            cx = np.sum(np.cos(c * x))
            dnx = x / nx
            dcx = -c * np.sin(c * x)
            return (a * b) / np.sqrt(d) * np.exp(-b * np.sqrt(nx / d)) * dnx - np.exp(cx / d) / d * dcx

    bounds = np.zeros((d, 2))
    bounds[:, 0] = -32.768
    bounds[:, 1] = 32.768
    xopt = [np.zeros(d)]

    return TestFunction(d, bounds, xopt, f, grad_f)


def test_six_hump():
    def f(xy):
        x = xy[0]
        y = xy[1]
        xterm = (4.0 - 2.1 * x**2 + x**4 / 3) * x**2
        yterm = (-4.0 + 4.0 * y**2) * y**2
        return xterm + x * y + yterm

    def grad_f(xy):
        x = xy[0]
        y = xy[1]
        dxterm = (-4.2 * x + 4.0 * x**3 / 3) * x**2 + (4.0 - 2.1 * x**2 + x**4 / 3) * 2.0 * x
        dyterm = (8.0 * y) * y**2 + (-4.0 + 4.0 * y**2) * 2.0 * y
        return np.array([dxterm + y, dyterm + x])

    xopt = [np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])]
    bounds = np.array([[-3.0, 3.0], [-2.0, 2.0]])
    
    return TestFunction(2, bounds, xopt, f, grad_f)


def test_branin(a=1.0, b=5.1 / (4 * np.pi**2), c=5 / np.pi, r=6.0, s=10.0, t=1.0 / (8 * np.pi)):
    def f(x):
        return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

    def grad_f(x):
        return np.array([
            2 * a * (x[1] - b * x[0]**2 + c * x[0] - r) * (-2 * b * x[0] + c) - s * (1 - t) * np.sin(x[0]),
            2 * a * (x[1] - b * x[0]**2 + c * x[0] - r)
        ])

    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    xopt = [np.array([-np.pi, 12.275]), np.array([np.pi, 2.275]), np.array([9.42478, 2.475])]
    
    return TestFunction(2, bounds, xopt, f, grad_f)


def test_gramacy_lee():
    def f(x):
        return np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1.0)**4

    def grad_f(x):
        return np.array([5 * np.pi * np.cos(10 * np.pi * x[0]) / x[0] - np.sin(10 * np.pi * x[0]) / (2 * x[0]**2) + 4 * (x[0] - 1.0)**3])

    bounds = np.zeros((1, 2))
    bounds[0, 0] = 0.5
    bounds[0, 1] = 2.5
    xopt = [np.array([0.548])]
    
    return TestFunction(1, bounds, xopt, f, grad_f)


def test_six_hump():
    def f(xy):
        x = xy[0]
        y = xy[1]
        xterm = (4.0 - 2.1 * x**2 + x**4 / 3) * x**2
        yterm = (-4.0 + 4.0 * y**2) * y**2
        return xterm + x * y + yterm

    def grad_f(xy):
        x = xy[0]
        y = xy[1]
        dxterm = (-4.2 * x + 4.0 * x**3 / 3) * x**2 + (4.0 - 2.1 * x**2 + x**4 / 3) * 2.0 * x
        dyterm = (8.0 * y) * y**2 + (-4.0 + 4.0 * y**2) * 2.0 * y
        return np.array([dxterm + y, dyterm + x])

    xopt = [np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])]
    bounds = np.array([[-3.0, 3.0], [-2.0, 2.0]])

    return TestFunction(2, bounds, xopt, f, grad_f)
