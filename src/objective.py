import numpy as np

class Objective:
    pass

class LinearNonnegative(Objective):
    def __init__(self, c):
        if not np.all(c > 0):
            raise ValueError("All elements must be strictly positive.")
        self.c = c

    def f(self, v):
        if np.all(self.c <= v):
            return 0
        return np.inf

    def grad(self, v):
        if np.all(self.c <= v):
            return np.zeros_like(v)
        return np.full_like(v, np.inf)

    def lower_limit(self):
        return self.c + 1e-8

    def upper_limit(self):
        return np.inf


class BasketLiquidation(Objective):
    def __init__(self, i, deltain):
        if i <= 0 or i > len(deltain):
            raise ValueError("Invalid index i.")
        self.i = i
        self.deltain = deltain

    def f(self, v):
        if v[self.i - 1] >= 1.0:
            return sum(0.0 if i == self.i - 1 else self.deltain[i] * v[i] for i in range(len(v)))
        return np.inf

    def grad(self, v):
        if v[self.i - 1] >= 1.0:
            g = self.deltain.copy()
            g[self.i - 1] = 0
            return g
        return np.full_like(v, np.inf)

    def lower_limit(self):
        ret = np.full(len(self.deltain), np.sqrt(np.finfo(float).eps))
        ret[self.i - 1] = 1.0 + np.sqrt(np.finfo(float).eps)
        return ret

    def upper_limit(self):
        return np.inf


class Swap(BasketLiquidation):
    def __init__(self, i, j, rho, n):
        deltain = np.zeros(n)
        deltain[j - 1] = rho
        super().__init__(i, deltain)
