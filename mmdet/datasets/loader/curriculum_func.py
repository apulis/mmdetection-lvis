import numpy as np


class CurriculumFunc(object):

    def __init__(self, func):
        func_names = [
            'const', 'linear', 'quadratic_convex', 'quadratic_concave',
            'cosine'
        ]
        assert func in func_names, '{} is not a valid function name.'
        self.func = eval('self.' + func)

    def const(self, phase):
        return 1.0

    def linear(self, phase):
        return phase

    def quadratic_convex(self, phase):
        return phase**2

    def quadratic_concave(self, phase):
        return 2 * phase - phase**2

    def cosine(self, phase):
        return (1 - np.cos(np.pi * phase))/2

    def __call__(self, phase):
        return self.func(phase)
