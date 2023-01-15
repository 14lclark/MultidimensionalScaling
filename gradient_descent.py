import numpy as np

# func must take numpy array as input
def gradient(func, pt, dx = 1e-6):
    grad = np.zeros(len(pt))
    for i in range(len(pt)):
        upper = pt.copy()
        lower = pt.copy()
        upper[i] += dx
        lower[i] -= dx
        grad[i] = (func(upper) - func(lower)) / (2*dx)
    return grad

def gradient_descent(
                    func, # function to optimize
                    init: np.ndarray, # starting point, np array
                    eta = lambda x: 1/x, # learning rate schedule
                    epsilon: np.double = 1e-10, # tolerance
                    max_iter: int = 1e5,
                    ):
    delta, iter_num = 1, 1
    pos = init
    curr = func(init)
    while (abs(delta) > epsilon or iter_num == 0) and iter_num < max_iter:
        last = curr
        print(last)
        # print(pos)
        pos = pos - eta(iter_num) * gradient(func, pos)
        # print(pos)
        curr = func(pos)
        print(curr)
        delta = curr - last
        print(delta)
        print(iter_num)
        iter_num += 1
        
    return pos

# Examples

def square(x):
     return np.sum(x ** 2)


print(gradient(square, np.array([2.,3.,4.])))

print(gradient_descent(square, np.array([254.,-13.,434.])))