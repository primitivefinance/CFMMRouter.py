import numpy as np

class LinearNonnegative:
    '''
    Objective function used for arbitrage detection. Whenever an arbitrage opportunity is found, the optimization problem yields a nonzero solution.
    --- parameters ---
    `c`:    Positive price vector.
    '''
    def __init__(self, c):
        if not np.all(c > 0):
            raise ValueError("All elements must be strictly positive.")
        self.c = c

    def f(self, v):
        # Evaluate the conjugate of the utility function of the `LineaNonnegative` objective at `v`.
        if np.all(self.c <= v):
            return 0
        return np.inf

    def grad(self, v):
        if np.all(self.c <= v):
            return np.zeros(len(self.c))
        output = []
        for i in range(len(self.c)):
            output.append(np.inf)
        return output

    def lower_limit(self):
        return self.c + 1e-8

    def upper_limit(self):
        output = []
        for i in range(len(self.c)):
            output.append(np.inf)
        return output

class BasketLiquidation:
    '''
    Objective function used for liquidating a basket of tokens to recieve maximum amount of token with index `i`.
    --- parameters ---
    `deltain`:  Vector of token balances before liquidation.
    '''
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
    '''
    Special case of basket liquiditation that swaps from token j to token i while maximizing the amount of token i recieved.
    --- parameters ---
    `i`:        Index of token to swap into.
    `j`:        Index of token to swap out of.
    `delta`:    Desired swap quantity.
    `n`:        Number of tokens.
    '''
    def __init__(self, i, j, delta, n):
        deltain = np.zeros(n)
        deltain[j-1] = delta
        super().__init__(i, deltain)
