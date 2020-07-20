import numpy as np
import ast

from autograd import grad
from utils import *


class DTMCImpl():
    def __init__(self):
        pass

    def __generate_x(self, shape, lower, upper):
        size = np.prod(shape)
        x = np.random.rand(size)

        x = (upper - lower) * x + lower

        return x

    def solve(self, model, assertion, display=None):
        spec = assertion
        lower = model.lower
        upper = model.upper
        x0 = np.array(ast.literal_eval(read(spec['x0'])))
        y0 = np.argmax(model.apply(x0), axis=1)[0]
        if 'fairness' in spec:
            sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            for index in range(x0.size):
                if not (index in sensitive):
                    lower[index] = x0[index]
                    upper[index] = x0[index]

        x = self.__generate_x(model.shape, lower, upper)
        y = np.argmax(model.apply(x), axis=1)[0]
